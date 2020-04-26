import os
import subprocess
import tempfile
import re
import nltk

punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']

def list2tree(node):
    if isinstance(node, list):
        tree = []
        for child in node:
            tree.append(list2tree(child))
        return nltk.Tree('NT', tree)
    elif isinstance(node, tuple):
        return nltk.Tree(node[1], [node[0]])

def process_tree(node):  # remove punct from tree
    if isinstance(node, str):
        return node
    label = node.label()
    if label in punctuation_tags:
        return None
    children = list()
    for child in node:
        proc_child = process_tree(child)
        if proc_child is not None:
            children.append(proc_child)
    if len(children) > 0:
        return nltk.Tree(label, children)
    else:
        return None


def evalb(pred_tree_list, targ_tree_list, evalb_dir='./EVALB'):

    temp_path = tempfile.TemporaryDirectory(prefix="evalb-")
    temp_file_path = os.path.join(temp_path.name, "pred_trees.txt")
    temp_targ_path = os.path.join(temp_path.name, "true_trees.txt")
    temp_eval_path = os.path.join(temp_path.name, "evals.txt")

    temp_tree_file = open(temp_file_path, "w")
    temp_targ_file = open(temp_targ_path, "w")

    for pred_tree, targ_tree in zip(pred_tree_list, targ_tree_list):
        temp_tree_file.write(re.sub('[ |\n]+', ' ', str(process_tree(list2tree(pred_tree)))) + '\n')
        temp_targ_file.write(re.sub('[ |\n]+', ' ', str(process_tree(targ_tree))) + '\n')
    
    temp_tree_file.close()
    temp_targ_file.close()

    evalb_param_path = os.path.join(evalb_dir, "fhs.prm")
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        temp_targ_path,
        temp_file_path,
        temp_eval_path)

    subprocess.run(command, shell=True)

    with open(temp_eval_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_fscore = float(match.group(1))
                break

    temp_path.cleanup()

    return evalb_fscore
