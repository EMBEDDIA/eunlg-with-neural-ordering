import logging
from typing import List
import os
from argparse import Namespace

from numpy.random.mtrand import RandomState
import torch

from core.models import Message, DocumentPlanNode
from core.pipeline import NLGPipelineComponent, LanguageSplitComponent
from core.realize_slots import SlotRealizer
from core.registry import Registry
from core.template_selector import TemplateSelector
from core.morphological_realizer import MorphologicalRealizer
from croatian_simple_morpological_realizer import CroatianSimpleMorphologicalRealizer
from english_uralicNLP_morphological_realizer import EnglishUralicNLPMorphologicalRealizer
from eu_date_realizer import EnglishEUDateRealizer, FinnishEUDateRealizer, CroatianEUDateRealizer, GermanEUDateRealizer
from eu_named_entity_resolver import EUEntityNameResolver
from eu_number_realizer import EUNumberRealizer
from finnish_uralicNLP_morphological_realizer import FinnishUralicNLPMorphologicalRealizer

import scorers.models
from scorers.sentence_encoder import SentenceEncoder


log = logging.getLogger("root")

# get device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# change path as desired
path = '~/'

MODEL_DIR = path + 'projects/embeddia/models/scorers'
WRKDIR = path + 'projects/embeddia/eunlg'


class EUNeuralSentenceOrderingScorer(NLGPipelineComponent):
    def run(
        self,
        registry: Registry,
        random: RandomState,
        language: str,
        core_messages: List[Message],
        expanded_messages: List[Message]
    ):
        """
        Runs this pipeline component.
        """

        template_selector = TemplateSelector()
        slot_realizer = SlotRealizer()
        date_realizer = LanguageSplitComponent(
            {
                "en": EnglishEUDateRealizer(),
                "fi": FinnishEUDateRealizer(),
                "hr": CroatianEUDateRealizer(),
                "de": GermanEUDateRealizer(),
            }
        )
        entity_name_resolver = EUEntityNameResolver()
        number_realizer = EUNumberRealizer()
        morphological_realizer = MorphologicalRealizer(
            {
                "en": EnglishUralicNLPMorphologicalRealizer(),
                "fi": FinnishUralicNLPMorphologicalRealizer(),
                "hr": CroatianSimpleMorphologicalRealizer(),
            }
        )
        old_log_level = log.level
        log.setLevel(logging.WARNING)

        for msg in core_messages:
            doc_plan = DocumentPlanNode([msg])
            template_selector.run(registry, random, language, doc_plan, core_messages)
            slot_realizer.run(registry, random, language, doc_plan)
            date_realizer.run(registry, random, language, doc_plan)
            entity_name_resolver.run(registry, random, language, doc_plan)
            number_realizer.run(registry, random, language, doc_plan)
            morphological_realizer.run(registry, random, language, doc_plan)

        log.setLevel(old_log_level)
        # turn msgs to lists of strings (tokens):
        msgs_as_strs = [[str(component.value) for component in msg.template.components] for msg in core_messages]

        # outdir of form <model_task+model_type+>_l<sequence_length>_<scale_min-scale_max>
        outdir = registry.outdir

        scorer_name, len_seq, scale = outdir.split('_')
        scorer_fname = scorer_name + '_{}.pt'.format(language[:2])  # language either 'en' or 'fi'

        len_seq = int(len_seq[1:])
        scale_min, scale_max = list(map(float, scale.split('-')))

        path = os.path.join(MODEL_DIR, scorer_fname)

        # model arguments are loaded from model file
        scorer_args = torch.load(path, map_location=DEVICE)
        emb_pars = scorer_args['emb_pars']

        # n_embs needed only when loading embeddings from file
        n_embs = len(msgs_as_strs) if emb_pars['type'] != 'tokens' else len([t for m in msgs_as_strs for t in m])
        assert 'model' in emb_pars

        if 'pw' in scorer_name:
            scorer_name = scorer_name.replace('pw+', 'Pairwise')
        elif 'ptr' in scorer_name:
            scorer_name = 'PointerNet'
        else:
            scorer_name = 'PositionNet'

        params = Namespace(**scorer_args)
        scorer = getattr(scorers.models, scorer_name)
        scorer = scorer(params=params, n_embs=n_embs)
        scorer.load_state_dict(scorer_args['model_state_dict'])
        scorer = scorer.to(DEVICE)
        scorer.eval()

        emb_pars['lang'] = language
        # initialise encoder model for encoding strings into embeddings
        encoder = SentenceEncoder(emb_pars=emb_pars, device=DEVICE)

        for i, (msg, msg_str) in enumerate(zip(core_messages, msgs_as_strs)):

            if len_seq != 'all':            # using n (e.g. 20) msgs around the given msg, other_msgs
                if params.task != 'order':
                    li, ri = max(0, i - len_seq // 2), min(len(msgs_as_strs), i + (len_seq - len_seq // 2))
                    ri = ri - (i - len_seq // 2) if (i - len_seq // 2) < 0 else ri
                    ri = len(msgs_as_strs) if ri > len(msgs_as_strs) else ri
                    other_msgs = msgs_as_strs[li:i] + msgs_as_strs[i+1:ri]

                else:
                    other_msgs = msgs_as_strs[i:i + len_seq] if i + len_seq <= len(msgs_as_strs) \
                        else [e for e in reversed(msgs_as_strs[i - len_seq + 1:i + 1])]
                seq_e = None    # getting seq_emb in sentence_encoder.py
            else:
                other_msgs = msgs_as_strs[:i] + msgs_as_strs[i + 1:]        # all except given msg
                seq_e = encoder.encode_seq(other_msgs, merge='avg') if params.task == 'pos' else None

            score = encode_and_score(scorer, msg_str, other_msgs, encoder, params, seq_emb=seq_e)
            coef = (score * (scale_max - scale_min) + scale_min)
            msg.score *= coef

        # Remember to undo the temporary template selection done above.
        for msg in core_messages:
            msg.template = None

        # print new order
        # tups = [(i, msg) for i, msg in enumerate(core_messages)]
        # core_messages = sorted(core_messages, key=lambda x: float(x.score), reverse=True)
        # tups = sorted(tups, key=lambda tup: float(tup[1].score), reverse=True)
        # print('New order: ', [tup[0] for tup in tups])

        return core_messages, expanded_messages


def encode_and_score(model, msg, other_msgs, encoder, pars, seq_emb=None):

    #  encode 20 msgs at a time, computation gets heavy with too many
    bs = 20
    r = len(other_msgs) % bs
    nb = len(other_msgs) // bs + int(bool(r))

    if seq_emb is not None:
        assert pars.task == 'pos'
        e = encoder.encode_msg(msg)
        e = torch.cat([e, seq_emb.unsqueeze(0).repeat(e.size(0), 1)], dim=1) \
            if encoder.emb_type == 'tokens' else torch.cat([e, seq_emb], dim=0)
    else:
        e = encoder.encode_msg(msg, seq=other_msgs) if pars.task == 'pos' else encoder.encode_msg(msg)

    score = 0
    for i in range(nb):
        if i + 1 == nb and r:
            b = other_msgs[i * bs: i * bs + r]
        else:
            b = other_msgs[i * bs: (i + 1) * bs]

        embs = [encoder.encode_msg(m) for m in b] if pars.task != 'pos' else None
        embs = torch.stack(embs) if pars.task == 'order' else embs
        score += compute_score(model, e, embs, pars.task).item()
    score /= nb
    return score


def compute_score(model, x, embs, model_type):
    """
    :param model:
    :param x:
    :param embs:
    :param model_type: pos / ptr / pw
    :return:
    """

    if model_type == 'pos':
        # position classifier
        # assuming pred is a Q-dimensional position distribution (softmax) vector
        pred = model(x).squeeze()                               # add batch dimension
        first_q, last_q = 1, len(pred)
        quantiles = torch.arange(start=first_q, end=last_q + 1, device=DEVICE)
        score = torch.sum(pred * quantiles, dim=0)              # in range [1, 10]
        score = 1 - (score - first_q) / (last_q - first_q)      # scale to [0, 1]: closer to 1 if score close to 1

    elif model_type == 'order':
        # get first predicted position, if correct -> score 1, if last -> score 0
        pred, _ = model(embs, torch.arange(end=len(embs)), teacher_force_ratio=0., device=DEVICE)
        pred_pos = pred[0]
        act_pos = 0         # if msg is the first element of embs
        score = 1 - (pred_pos - act_pos) / float(len(embs))

    else:
        # pairwise
        score = 0
        for e in embs:
            pred = model(x, e)
            score += torch.log(pred)

        score /= len(embs)              # average
        score = torch.exp(score)        # turn score into interval of [0, 1]

    return score
