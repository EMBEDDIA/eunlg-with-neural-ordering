from pathlib import Path
import argparse
import numpy as np
from pandas import DataFrame

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out', nargs='?')
parser.add_argument('-bl', '--baselines', action='store_true')
args = parser.parse_args()

baselines = ["baseline_filter", "baseline_filter_ctx", "baseline_filter_ctx_setpen", "baseline_filter_setpen"]
neurals = ["neural_filter", "neural_filter_ctx", "neural_filter_ctx_setpen", "neural_filter_setpen"]

variants = baselines if args.baselines else neurals

data_multi = DataFrame(columns=["variant", "dataset", "location", "text"])
data_single = DataFrame(columns=["variant", "dataset", "location", "text"])

for variant_file in (Path(__file__).parent / args.out).glob("*"):
    if not variant_file.is_file:
        continue
    if not variant_file.name.endswith(".txt"):
        continue
    text_id = str(variant_file.name[:-4])

    variant, language, dataset, location = text_id.split("-")
    if language != "en":
        continue
    # if variant not in ["baseline_filter", "neural_filter", "neural_filter_ctx", "neural_filter_ctx_setpen"]:
    if variant not in variants:
        continue

    text = variant_file.read_text()
    if location == "all":
        data_multi.loc[text_id] = [variant] + [dataset] + [location] + [text]
    else:
        data_single.loc[text_id] = [variant] + [dataset] + [location] + [text]

file_content_prefix = """
<html>
<head>
<style>
    html {
        font-size: 6pt;
    }
    table {
        border-spacing: 0 2em;
    }
    @media print {
        table {
            page-break-after:always
        }
        body {
            margin-top: -.9cm:
            margin-bottom: -2cm;
            margin-left: -1cm;
            margin-right: 0cm;
        }
    }
    tr {
        border-bottom: 1px solid black;
        border-top: 1px solid black;
        margin-bottom: 4em;
        max-height: 90%;
        overflow-y: hidden;
    }
    td {
        vertical-align: top;
        padding: 1em;
        font-size: 0.7em;
    }
    th {
        vertical-align: middle;
        font-size: 1em;
    }
    .head-row {
        margin-bottom: 0;
    }
    .text {
        margin-top: 0;
        width: 20%;
        max-height: 10cm;
        overflow-y: hidden;
    }
    .row-head {
        width: 12%;
    }
</style>
</head>
<body>
"""

file_content_suffix = """
</body>
</html>
"""

data = data_multi


grps = data_single.groupby(["location", "dataset"])
rng = np.arange(grps.ngroups)
np.random.shuffle(rng)
data = data.append(data_single[grps.ngroup().isin(rng[:100])])

print(data)

with open("side_by_side.html", "w") as out_file:
    out_file.write(file_content_prefix)
    for (dataset, location), grp in data.groupby(["dataset", "location"]):
        out_file.write(
            # f'<table><tr class="head"><th></th><th>baseline</th><th>score</th><th>score_filter</th><th>full</th></tr>'
            # f'<table><tr class="head"><th></th><th>baseline_filter</th><th>neural_filter</th>' + \
            #     '<th>neural_filter_ctx</th><th>neural_filter_ctx_setpen</th></tr>'
            f'<table><tr class="head"><th></th><th>{variants[0]}</th><th>{variants[1]}</th>' + \
            f'<th>{variants[2]}</th><th>{variants[3]}</th></tr>'
            # noqa: E501
        )
        out_file.write(f'<tr><th class="row-head">{dataset}<br>{location}</th>')
        # for variant in ["baseline", "score", "score_filter", "full"]:
        # for variant in ["baseline_filter", "neural_filter", "neural_filter_ctx", "neural_filter_ctx_setpen"]:
        for variant in variants:
            for _, row in grp.iterrows():
                if row["variant"] == variant:
                    out_file.write(f'<td class="text">{row["text"]}</td>')
        out_file.write("</tr></table>")
    out_file.write(file_content_suffix)
