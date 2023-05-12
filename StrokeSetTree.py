class StrokeTree:
    def __init__(self, input_strokesets):
        self.root = StrokeNode(None)
        self.create_tree(input_strokesets)
    
    def create_tree(self, input_strokesets):
        stack = [(self.root, input_strokesets)]
        while stack:
            parent, strokesets = stack.pop()
            for strokeset in strokesets:
                strokeset_node = StrokeNode(strokeset, parent)
                if len(strokeset) > 1:
                    children = self.split_into_children(strokeset)
                    stack.append((strokeset_node, children))
        
    def split_into_children(self, strokeset):
        # split strokeset into child strokesets based on distance between strokes
        children = []
        # ...
        return children
    
    def get_datapoints(self, level):
        datapoints = []
        self.traverse(self.root, datapoints, 0, level)
        return datapoints
    
    def traverse(self, node, datapoints, current_level, target_level):
        if current_level == target_level:
            if not node.children:
                datapoints.append(node.strokes)
        else:
            for child in node.children:
                self.traverse(child, datapoints, current_level+1, target_level)


class StrokeNode:
    def __init__(self, strokes, parent=None):
        self.strokes = strokes
        self.parent = parent
        self.children = []
        if parent:
            parent.add_child(self)
        
    def add_child(self, node):
        self.children.append(node)
