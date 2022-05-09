# Engineering Notebook

### May 8

##### Morning

I've read through the codebase of FewX and compared it with TFA's FS-Det and Detectron2. What turns out happening is that FewX rewrote a certain chunk of the data-loading code which TFA simply took from Detectron2. In particular, FewX has a custom SupportDatasetMapper, which loads in the support information/images from the start so that during training, it simply samples a support image for each of the query image. This is obviously easily extensible to a Two-Way Contrastive Training strategy. FewX implements Two-Way Contrastive Training by selecting a class from the query image to be **the class**. Then selecting a positive support of the same class. Then, selecting a negative support of a class that **does not exist** in the query image. The gt_boxes of query can be filtered at anytime, it doesn't matter. I think that's what I would do except I would possibly store the support information in a more json format.

##### Evening

I've read through the inference part of the codebase of FewX. It turns out that for dataloading of the training, FewX just loads a Pandas Dataframe in using a DatasetMapper and then randomly pool support images for each of the query (all gt_boxes of the same class in the query is used when judging foreground to background in the RPN). For inference, FewX loads into the model via a `model_init` function the support images and then preprocess their features. These support images is stored in a 10-shot Pandas Dataframe with the same structure as all the other supports (which is in a separate dataframe). The query is compared to the average of each support class's images.

### May 9

##### Early Morning

I read through the dataloading portion and some modeling portion of TFA's FS-Det, which I shall call FS-Det henceforth. Again, FS-Det is finetuning based so there's no support-class messiness. The Pascal few shot splits are stored simply as a `.txt` file recording several images/xml labels that contain the corresponding class. Inside the data-loading code (at the registration of the dataset), we load in 10 instances (for 10-shot) for each few shot class from a certain split. Each split is registered as a different dataset. The way we load in 10 instances is to separate the consideration of instance from image (so a couple instances can be from the same image, different from FewX's training sampling). For each image we consider (there may be repeats), we only retain the bounding box for one instance. This is all done in the dataset level, without any surgery on the DatasetMapper level because the format is the standard one. So how doesn't only considering one instance affect the training? The reason is because FS-Det doesn't finetune the RPN, so the lack of other instances/boxes will not discourage the RPN from recognizing them. Each proposal output from the RPN **during training** is necessarily a **foreground** associated with a gt_box (backgrounds are thrown away). So, our instance also only affects the RoI head's learning on that instance and not others. Thus this is a valid structure for the purpose of FS-Det. 

The problem with FS-Det's dataloading is two-folded. 1. it doesn't load support classes; 2. it doesn't load all instances of the support class from a query image, which is required for the RPN.

