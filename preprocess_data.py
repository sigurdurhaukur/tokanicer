import os
import xml.etree.ElementTree as ET
import re


class XML_file:
    def __init__(self, path):
        self.path = path
        self.tree = ET.parse(path, parser=ET.XMLParser(encoding="utf-8"))
        self.root = self.tree.getroot()

    def read_xml(self):
        try:

            extracted_text = []

            # Define the namespace
            namespaces = {"tei": "http://www.tei-c.org/ns/1.0"}

            # specific to the IGC-News1-22.10.TEI dataset
            # special tokens are wrapped in <> and are used to denote the start and end of a special token
            for term in self.root.findall(
                ".//tei:profileDesc//tei:textClass//tei:keywords//tei:term", namespaces
            ):
                if term.text:
                    term.text = "<" + term.text.strip() + ">"
                    extracted_text.append(term.text.strip() + " ")

            # Iterate over all <p> elements and collect their text content
            for elem in self.root.findall(".//tei:p", namespaces):
                elem.text = (
                    elem.text.replace("Þessi texti er gefinn út með", "")
                    .replace("This work is licensed under the", "")
                    .replace(
                        'prefix = o ns = "urn:schemas-microsoft-com:office:office" /',
                        "",
                    )
                    .replace("xml:namespace", "")
                )
                if elem.text is not None and elem.text.strip() != "":
                    extracted_text.append(elem.text + "\n\n")

            # Join all pieces of text into a single string
            extracted_text = "".join(filter(None, extracted_text))

            return extracted_text

            # text = []
            # for elem in self.root.iter():
            #     inner_text = elem.text
            #     if inner_text is not None:

            #         inner_text = inner_text.replace(
            #             'xml:namespace prefix = o ns = "urn:schemas-microsoft-com:office:office" /',
            #             "",
            #         )

            #         text.append(inner_text.strip())

            #         # conserve newlines
            #         if elem.tag == "p":
            #             text.append("\n")

            # extracted_text = " ".join(text)

            # prescript_len = len(extracted_text.split("      ")[0])
            # extracted_text = extracted_text[prescript_len:].lstrip()

            # extracted_text += "\n"

            # return extracted_text
        except Exception as e:
            print("Error: " + str(e))
            return ""


def collect_all_paths_in_dir(path, max_paths=None):
    all_paths = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".xml"):
                all_paths.append(os.path.join(root, file))

            if max_paths is not None and len(all_paths) == max_paths:
                break

    if max_paths is None:
        return all_paths
    else:
        return all_paths[:max_paths]


def save(path, data):
    with open(path, "w+") as f:
        f.write(data)


def clean_dir(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))


if __name__ == "__main__":
    print("Collecting all paths...")
    save_dir = "processed_data/"
    all_paths = collect_all_paths_in_dir("raw_data/", max_paths=150)
    # all_paths = [
    #     "./raw_data/IGC-News1-22.10.TEI/frettabladid_is/2021/10/IGC-News1-frettabladidis_8501947.xml"
    # ] # for testing

    if os.path.exists(save_dir):
        clean_dir(save_dir)
        print(f"Cleaned {save_dir}")
    else:
        os.mkdir(save_dir)
        print(f"Created {save_dir}")

    print("Processing...")
    for i, path in enumerate(all_paths):
        try:
            data = XML_file(path).read_xml()
            new_path = save_dir + str(i) + ".txt"
            save(new_path, data)

            print(
                f"Processed:  {i} / {len(all_paths)} total left: {len(all_paths) - i}"
            )
        except Exception as e:
            print("could not process file: " + path)
            print("Error: " + str(e))

    print("Done!")
