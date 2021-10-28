from pathlib import Path
from pandas import DataFrame
import csv

multi_loc = DataFrame(columns=["variant", "dataset", "location", "para1", "para2", "para3"])
single_loc = DataFrame(columns=["variant", "dataset", "location", "para1", "para2", "para3"])

for variant_file in (Path(__file__).parent / "bulk_out").glob("*"):
    if not variant_file.is_file:
        continue
    if not variant_file.name.endswith(".txt"):
        continue
    text_id = str(variant_file.name[:-4])

    variant, language, dataset, location = text_id.split("-")
    if language != "en":
        continue
    if variant not in ["random", "baseline", "score", "full"]:
        continue

    text = [" ", " ", " "]
    idx = 0
    for para in variant_file.read_text().split("<p>"):
        para = para.replace("<p>", "")
        para = para.replace("</p>", "")
        para = para.replace("\n", "")
        para = para.strip()
        if para:
            text[idx] = para
            idx += 1

    if "all" in text_id:
        multi_loc.loc[text_id] = [variant] + [dataset] + [location] + text
    else:
        single_loc.loc[text_id] = [variant] + [dataset] + [location] + text


sample_dataset_and_location = (
    single_loc.groupby(["variant", "dataset"])
    .sample(2)
    .query("variant == 'full'")[["dataset", "location"]]
    .drop_duplicates(ignore_index=True)
)
sampled = single_loc[
    single_loc[["dataset", "location"]].apply(tuple, 1).isin(sample_dataset_and_location.apply(tuple, 1))
]
print(sampled)
# sampled[["para1", "para2", "para3"]].to_csv('single_loc.csv', quoting=csv.QUOTE_NONNUMERIC, index_label="variant")


multi_loc[["para1", "para2", "para3"]].to_csv("multi_loc.csv", quoting=csv.QUOTE_NONNUMERIC, index_label="variant")
