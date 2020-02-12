
class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    """
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_classs_ids = {}
    
    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"

        for info in self.class_info:
            if info["source"] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({"source": source,
                                "id": class_id,
                                "name": class_name})
    
    def add_image(self, source, image_id, path, **kwargs):
        image_info = {"id": image_id,
                    "source": source,
                    "path": path}
        image_info.update(kwargs)
        self.image_info.append(image_info)
    
    def prepare(self):
        ''' 
            prepare the dataset class for use
        '''
        def clean_name(name):
            return  ",".join(name.split(",")[:1])
            
        self.num_of_class = len(self.class_info)
        self.class_ids = np.arange(self.num_of_class)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info["source"], info["id"]): id
                                                for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info["source"], info["id"]): id
                                                for info, id in zip(self.image_info, self.image_ids)}
        self.sources = list(set([i["source"] for i in self.class_info]))

        self.source_class_ids = {}
        for source in self.sources:
            self.source_class_ids[source] = []
            for i, info in enumerate(self.class_info):
                if i==0 or source==info["source"]:
                    self.source_class_ids[source].append(i)
        

        
    def map_source_class_id(self, source_class_id):
        ''' Take a source class ID and return the int class ID assigned to it

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        '''
        return self.class_from_source_map[source_class_id]

    # def

    

        