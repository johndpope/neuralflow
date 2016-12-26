import pickle

from neuralflow.utils.LatexExporter import LatexExporter


def export_results(result_list, output_dir):
    exp = LatexExporter(filename=output_dir + "results")
    exp.export(result_list)
    pickle.dump(result_list, open(output_dir + "results.pkl", "wb"))