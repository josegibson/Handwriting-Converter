import xml.etree.ElementTree as ET


def print_tree(elem, level=0):
    tags_count = {}
    for child in elem:
        if child.tag not in tags_count:
            tags_count[child.tag] = 1
        else:
            tags_count[child.tag] += 1
        
        if tags_count[child.tag] <= 2:
            print('\t' * level +  '<' + child.tag + '>')
            if child.text and child.text.strip():
                text = child.text.strip()
                text = text.replace('\n', '\n' + '\t' * (level + 1))
                print('\t' * (level + 1) + text)
            print_tree(child, level + 1)
            if child.tail and child.tail.strip():
                print(' ' * 4 * level + child.tail.strip())
        else:
            if tags_count[child.tag] == 3:
                print('\t' * level + '<' + child.tag + '>')
                print('\t' * (level + 1) + '...')
                if child.tail and child.tail.strip():
                    print('\t' * level + child.tail.strip())
    return


tree = ET.parse('original/a01/a01-001/strokesz.xml')
root = tree.getroot()
print_tree(root)
