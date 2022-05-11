from tree_sitter import Language, Parser
import json

JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
KOTLIN_LANGUAGE = Language('build/my-languages.so', 'kotlin')
CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

with open('parsed_datasets/output_labels.json') as json_file:
    output_labels = json.load(json_file)

# Java
java_parser = Parser()
java_parser.set_language(JAVA_LANGUAGE)

dataset = open("../data/codeforces_tags_java.csv", "r")
parsed_dataset = open("parsed_datasets/java.txt", "w")
for line in dataset:
    if line == None or len(line) <= 1:
        break
    reversed_line = line[::-1]
    idx = find_nth(reversed_line, '"', 7)
    idx2 = find_nth(reversed_line, '"', 5)
    idx, idx2 = len(line) - idx - 1, len(line) - idx2 - 1
    code = line[1:idx]
    code = code.replace('\\n', '\n').replace('\\t', '\t')
    classes = line[idx+3:idx2]
    labels = classes.split(",")
    label_assignments = set()
    for label in labels:
        if label in output_labels:
            label_assignments.add(output_labels[label])
    labels_string = ""
    for num in label_assignments:
        if labels_string == "":
            labels_string = str(num)
        else:
            labels_string += ("," + str(num))
    
    tree = java_parser.parse(bytes(code, "utf-8"))
    parsed_code = tree.root_node.sexp()

    parsed_dataset.write(labels_string)
    parsed_dataset.write("\t\t\t\t\t\t")
    parsed_dataset.write(parsed_code)
    parsed_dataset.write("\n")
dataset.close()
parsed_dataset.close()

# Kotlin
kotlin_parser = Parser()
kotlin_parser.set_language(KOTLIN_LANGUAGE)

dataset = open("../data/codeforces_tags_kotlin.csv", "r")
parsed_dataset = open("parsed_datasets/kotlin.txt", "w")
for line in dataset:
    if line == None or len(line) <= 1:
        break
    reversed_line = line[::-1]
    idx = find_nth(reversed_line, '"', 7)
    idx2 = find_nth(reversed_line, '"', 5)
    idx, idx2 = len(line) - idx - 1, len(line) - idx2 - 1
    code = line[1:idx]
    code = code.replace('\\n', '\n').replace('\\t', '\t')
    classes = line[idx+3:idx2]
    labels = classes.split(",")
    label_assignments = set()
    for label in labels:
        if label in output_labels:
            label_assignments.add(output_labels[label])
    labels_string = ""
    for num in label_assignments:
        if labels_string == "":
            labels_string = str(num)
        else:
            labels_string += ("," + str(num))
    
    tree = kotlin_parser.parse(bytes(code, "utf-8"))
    parsed_code = tree.root_node.sexp()

    parsed_dataset.write(labels_string)
    parsed_dataset.write("\t\t\t\t\t\t")
    parsed_dataset.write(parsed_code)
    parsed_dataset.write("\n")
dataset.close()
parsed_dataset.close()

# C++
cpp_parser = Parser()
cpp_parser.set_language(CPP_LANGUAGE)

dataset = open("../data/codeforces_tags_cpp_large.csv", "r")
parsed_dataset = open("parsed_datasets/cpp.txt", "w")
for line in dataset:
    if line == None or len(line) <= 1:
        break
    reversed_line = line[::-1]
    idx = find_nth(reversed_line, '"', 7)
    idx2 = find_nth(reversed_line, '"', 5)
    idx, idx2 = len(line) - idx - 1, len(line) - idx2 - 1
    code = line[1:idx]
    code = code.replace('\\n', '\n').replace('\\t', '\t')
    classes = line[idx+3:idx2]
    labels = classes.split(",")
    label_assignments = set()
    for label in labels:
        if label in output_labels:
            label_assignments.add(output_labels[label])
    labels_string = ""
    for num in label_assignments:
        if labels_string == "":
            labels_string = str(num)
        else:
            labels_string += ("," + str(num))
    
    tree = cpp_parser.parse(bytes(code, "utf-8"))
    parsed_code = tree.root_node.sexp()

    parsed_dataset.write(labels_string)
    parsed_dataset.write("\t\t\t\t\t\t")
    parsed_dataset.write(parsed_code)
    parsed_dataset.write("\n")
dataset.close()
parsed_dataset.close()