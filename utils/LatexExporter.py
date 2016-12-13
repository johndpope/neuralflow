from typing import List


class LatexExporter:
    document_header = "\\documentclass{report}\n\\usepackage[utf8]{inputenc}\n\\usepackage{array}\n\\usepackage{makecell}\n\\newcolumntype" \
                      "{C}[1]{ > {\\centering\\arraybackslash}p{#1}}\n\\begin{document}\n"

    table_header = "\t\\begin{table}[!h]\n\t\\centering\n"

    footer = "\t\t\\hline\n\t\t\\end{tabular}\n\\end{table}\n\\end{document}"

    # hspace = "C{2cm}"
    hspace = " c "

    def __init__(self, filename):
        self.__filename = filename + ".tex"

    def __format_row(self, values):
        s = ["\t\t"]
        for i, v in enumerate(values):
            sep = "&" if i < len(values)-1 else ""
            s.append("{}{} ".format(str(v).replace("_", "\_"), sep))
        s.append("\\\\\n")
        return "".join(s)

    def export(self, row_dicts: List[dict]):
        out_file = open(self.__filename, "w")
        out_file.write(LatexExporter.document_header)
        out_file.write(LatexExporter.table_header)

        keys = row_dicts[0].keys()
        n_cols = len(keys)

        table_header = "\t\\begin{tabular}{|" + LatexExporter.hspace * n_cols + "|}\n\t\t\\hline\n"
        out_file.write(table_header)
        out_file.write(self.__format_row(keys))
        out_file.write("\t\t\\hline\n")

        for row in row_dicts:
            values = [row[k] for k in keys]
            out_file.write(self.__format_row(values))

        out_file.write(LatexExporter.footer)
        out_file.close()
