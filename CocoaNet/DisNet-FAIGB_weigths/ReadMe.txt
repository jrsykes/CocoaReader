Sweep to find the best architecture values and subsequent weights for DisNet RGB FAIGB photos. 
Trained on CSGPU5

The dataset can be found here: dat/FAIGB/FAIGB_FinalSplit_700
This dataset is composed of approx. 31,452 images of 107 classes of diseased and healthy planty species common to agriculture and forestry. 
This dataset was scraoped from the web using Google and Bing and two custom web scrapers.
The images were subsequently filtered using a pretrained Plant-NotPlant CNN and then hand filtered.
The images were also precompressed to 700x700 pixels. The full resolution dataset can be found here: dat/FAIGB/FAIGB_FinalSplit

Sweep results found here: