from pathlib import Path
import numpy as np
from pandas import DataFrame
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out', nargs='?')
args = parser.parse_args()

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
    if variant not in ["list_baseline", "list_neural"]:
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
data = data.append(data_single[grps.ngroup().isin(rng[:10])])

print(data)

with open("side_by_side_lists.html", "w") as out_file:
    out_file.write(file_content_prefix)
    for (dataset, location), grp in data.groupby(["dataset", "location"]):
        out_file.write(
            f'<table><tr class="head"><th></th><th>list_baseline</th><th>list_neural</th></tr>'  # noqa: E501
        )
        out_file.write(f'<tr><th class="row-head">{dataset}<br>{location}</th>')
        for variant in ["list_baseline", "list_neural"]:
            for _, row in grp.iterrows():
                if row["variant"] == variant:
                    out_file.write(f'<td class="text">{row["text"]}</td>')
        out_file.write("</tr></table>")
    out_file.write(file_content_suffix)
