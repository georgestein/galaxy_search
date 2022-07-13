# Welcome to Galaxy Finder!

This is an interactive galaxy visualization app using the [streamlit library](https://docs.streamlit.io/en/stable/):

### Check out the live version here: [share.streamlit.io/georgestein/galaxy_search](https://share.streamlit.io/georgestein/galaxy_search)

### Or find the training codes and data here: [github.com/georgestein/ssl-legacysurvey](https://github.com/georgestein/ssl-legacysurvey)


How to interact with this app:

1. Enter the coordinates of your favourite galaxy, or let it select a random one for you.
2. It searches for the most similar looking ones in the brightest 42 million galaxies [DECaLS dr9 dataset](https://www.legacysurvey.org/viewer).
3. It displays the images and datatables, and gives download links.
4. A variety of display options can be found in the sidebar.

Please use this app for any scientific investigations that you desire! Applications I can think of include discovering rare objects given only a single example, flagging and identifying bad data, and rapidly constructing and improving training sets for supervised applications - but I am sure there are many more!

![alt text](image.png)

---
  
### A bit about the method: 
- The similarity of two images is quite easy to judge by eye - but writing an algorithm to do the same is not as easy as one might think! This is because as hunans we can easily identify and understand what object is in the image.                     
- A machine is different - it simply looks individual pixel values. Yet two images that to us have very similar properties and appearences will likely have vastly different pixel values. For example, imagine rotating a galaxy image by 90 degrees. It it obviously still the same galaxy, but the pixel values have completeley changed.                                       
- So the first step is to teach a computer to understand what is actually in the image on a deeper level than just looking at pixel values. Unfortunately we do not have any information alongside the image specifying what type of galaxy is actually in it - so where do we start?                                                                                                   
- We used a type of machine learning called "self-supervised representation learning" to boil down each image into a concentrated vector of information, or "representation", that encapsulates the appearance and properties of the galaxy.
- Self-supervised learning works by creating multiple versions of each image which approximate the observational symmetries, errors, and uncertainties within the dataset, such as image rotations, adding noise, blurring it, etc., and then teaching the machine to learn the same representation for all these versions of the same galaxy. In this way, we move beyond looking at pixel values, and teach the machine a deeper understanding of the image.
- Once we have trained the machine learning model on millions of galaxies we calculate and save the representation of every image in the dataset, and precompute the similarity of any two galaxies. Then, you tell us what galaxy to use as a starting point, we find the representation belonging to the image of that galaxy, compare it to millions of other representations from all the other galaxies, and return the most similar images!

**Please see [our overview paper](https://arxiv.org/abs/2110.13151) for more technical details, or see our recent application of the app to find [strong gravitational lenses](https://arxiv.org/abs/2012.13083) -- some of the rarest and most interesting objects in the universe!**

Dataset:

- We used galaxy images from [DECaLS DR9](https://www.legacysurvey.org/), randomly sampling 3.5 million galaxies to train the machine learning model. We then apply it on every galaxy in the dataset, about 42 million galaxies with z-band magnitude < 20, so most bright things in the sky should be included, with very dim and small objects likely missing - more to come soon!                                                    
- The models were trained using images of size 96 pixels by 96 pixels centered on the galaxy. So features outside of this central region are not used to calculate the similarity, but are sometimes nice to look at                                                                                                         
                                                                                                             
Please note this project is ongoing, and results will continue to be updated and improved.           
                                                                                                             Created by [George Stein](https://georgestein.github.io/)                      
 