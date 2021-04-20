import streamlit as st
import numpy as np
import os
import sys

import time
import math
import pandas as pd
import urllib

from utils import *

class LoadCatalogue:
    """Separate loading from catalogue operations to use caching"""
    def __init__(self, data_loc='data/'):

        self.running_local = os.popen('hostname').read().startswith('Georges-MacBook-Pro')#.local'
        if self.running_local:
            self.data_loc = data_loc

        else:
            self.data_loc = 'https://portal.nersc.gov/project/nyx/decals_self_supervised/streamlit_app_data/data/'
            self.data_loc_local = 'data/'
             
    def get_local_or_url(self, file_in, check_fullsize=False, fullsize=None):
        """file_in can be either local destination or url"""
        if self.running_local:

            return file_in

        else:
            filepath = os.path.join(self.data_loc_local, os.path.basename(file_in)) # take file name from url
            if not os.path.exists(filepath) or check_fullsize:

                filesize_local = 0

                if os.path.exists(filepath):
                    fileinfo = os.stat(filepath)
                    filesize_local = fileinfo.st_size

                if fullsize != filesize_local:
                    lab = "Downloading {:s}... ".format(filepath)
                    if os.path.basename(file_in) == 'representations.npy':
                        lab += 'This one may take a while (up to ~5 mins), please stand by!\nOnce downloaded subsequent runs will be fast'

                    with st.spinner(lab):
                        urllib.request.urlretrieve(file_in, filepath)

            return filepath
            
    # @st.cache(allow_output_mutation=True)# #(suppress_st_warning=True)
    # cache works on local version, but not when deployed to share.streamlit.io
    # due to incorrect dictionary caching? Unclear...
    # @st.cache(persist=True, max_entries=1, allow_output_mutation=True, ttl=3600, hash_funcs={dict: lambda _: None})# 
    def load_catalogue_coordinates(self, extra_features=False,
                                   features_extra=['flux', 'z_phot_median', 'brickid',
                                                   'inds', 'objid', 'source_type', 'ebv']):

#        if not self.running_local:
#            st.write('Needs to retreive a few large files the first time you run it - please stand by!')

        self.features_extra = features_extra

        full_catalogue = {}
        
        # always load in ra and dec
        self.features_radec = ['ra', 'dec']
        for fstr in self.features_radec:
            full_catalogue[fstr] = np.load(self.get_local_or_url(os.path.join(self.data_loc, fstr+'.npy')))

        # option to add in additional catalogue features of interest
        if extra_features:
            # download files to use later
            for fstr in self.features_extra:
                self.get_local_or_url(os.path.join(self.data_loc, fstr+'.npy'))

        return full_catalogue

    # cache works on local version, but not when deployed to share.streamlit.io
    # due to incorrect dictionary caching? Unclear...
    # @st.cache(persist=True, max_entries=1, allow_output_mutation=True, ttl=3600, hash_funcs={dict: lambda _: None})# #(suppress_st_warning=True)
    def load_representations(self):
        """Keep seperate from loading in catalogues, as when representation file starts to get large will need to add in chunked access"""
        representations = np.load(self.get_local_or_url(os.path.join(self.data_loc, 'representations.npy'), check_fullsize=True, fullsize=224000128))

        return representations
    

class Catalogue:
    def __init__(self, full_catalogue, representations,
                 data_loc='data/'): 


        self.full_catalogue = full_catalogue
        self.representations = representations
        
        self.data_loc = data_loc
        
        self.pixel_size = 0.262 / 3600 # arcsec to degrees
    

    def load_from_catalogue_indices(self, inds_load=None, extra_features=True,
                                    features_extra=['flux', 'z_phot_median', 'brickid',
                                                    'inds', 'objid', 'source_type', 'ebv']):
        
        if inds_load is None:
            inds_load=self.similar_inds
            
        self.features_extra = features_extra
        
        similarity_catalogue = {}
        
        # always load in ra and dec
        self.features_radec = ['ra', 'dec']
        for fstr in self.features_radec:
            similarity_catalogue[fstr] = np.load(os.path.join(self.data_loc, fstr+'.npy'), mmap_mode='r')[inds_load]

        # option to add in additional catalogue features of interest
        if extra_features:
            for fstr in self.features_extra:
                similarity_catalogue[fstr] = np.load(os.path.join(self.data_loc, fstr+'.npy'), mmap_mode='r')[inds_load]

        return similarity_catalogue   
    
    
    def search_catalogue(self, ra, dec, nnearest=1, far_distance_npix=10):
        """Return index and ra dec of nearest galaxy to search point (query_ra, query_dec)"""
        
        self.search_ra = ra
        self.search_dec = dec
      
        # calculate angular seperation of all objects from query point
        sep = angular_separation(self.search_ra, self.search_dec, self.full_catalogue['ra'], self.full_catalogue['dec'])
        
        self.query_ind = np.argmin(sep)
        self.query_ra  = self.full_catalogue['ra'][self.query_ind]
        self.query_dec  = self.full_catalogue['dec'][self.query_ind]

        self.query_distance = sep[self.query_ind]
        
        if self.query_distance > far_distance_npix*self.pixel_size:
            # notify of bad query
            st.write(('\nClosest galaxy in catalogue is quite far away from search point ({:.2f} degrees).'
                      ' Either this galaxy is not yet in our database, or is not in the DECaLS dr9 footprint!\n'.format(self.query_distance)))

        del sep

    def similarity_search(self, nnearest=5, min_angular_separation=5, similarity_inv=False):
        """Return indices and similarity scores to nearest nsimilar data samples.
        First index returned is the query galaxy
        
        Parameters
        ----------
        nnearest: number of most similar galaxies to return
        min_angular_separation: minimum angular seperation of galaxies in pixelsize. Anything below is thrown out
        similarity_inv: if true returns most similar, if false returns least similar
        """
    
        nnearest_intermediate = int(nnearest*1.5) # some may be thrown out due to angular seperation constraints, so oversample
        self.similar_inds, self.similarity_score = calculate_similarity(self.representations, self.query_ind, nnearest=nnearest_intermediate, similarity_inv=similarity_inv)

        if similarity_inv:
            # append query to start of list, as it will no longer be most similar to itself
            self.similar_inds = np.insert(self.similar_inds, 0, self.query_ind)
            self.similarity_score = np.insert(self.similarity_score, 0, 1.)

        # now remove galaxies that are suspiciously close to each other on the sky
        # which happens when individual galaxies in a cluster are included as separate sources in the catalogue
        ra_similar = self.full_catalogue['ra'][self.similar_inds]
        dec_similar = self.full_catalogue['dec'][self.similar_inds]

        # all vs all calculation
        sep = angular_separation(ra_similar[np.newaxis, ...], dec_similar[np.newaxis, ...], ra_similar[..., np.newaxis], dec_similar[..., np.newaxis])

        # compile indices of galaxies too close in angular coordinates
        inds_del = set()
        for i in range(sep.shape[0]):
            inds_cuti = set(np.where(sep[i, (i+1):] < min_angular_separation*self.pixel_size)[0]+(i+1))
            inds_del  = inds_del | inds_cuti # keep only unique indices
            
        # remove duplicate galaxies from similarity arrays
        inds_del = sorted(inds_del) 
        self.similar_inds = np.delete(self.similar_inds, inds_del)
        self.similarity_score = np.delete(self.similarity_score, inds_del)

        self.similar_inds = self.similar_inds[:nnearest]
        self.similarity_score = self.similarity_score[:nnearest]

        
def main():

    st.title("Welcome to Galaxy Finder")
    #st.subheader("Enter the coordinates of your favourite galaxy and we'll search for the most similar looking ones in the universe!")
    #st.write("")
    #st.write("Click the 'search random galaxy' on the left for a new galaxy, or try finding a cool galaxy at https://www.legacysurvey.org/viewer")
    #st.write("Use the south survey (select the <Legacy Surveys DR9-south images> option)")
    #st.write("Please note products here are just initial trials, with small models that fit within the memory limits of streamlit.")

    with st.beta_expander('Instructions'):
        st.markdown(
            """
            **Enter the coordinates of your favourite galaxy and we'll search for the most similar looking ones in the universe!**
            
            Click the 'search random galaxy' on the left for a new galaxy, or try finding a cool galaxy at [legacysurvey.org](https://www.legacysurvey.org/viewer)
            - Use the south survey (select the <Legacy Surveys DR9-south images> option). Currently not all galaxies are included, but most bright ones should be.
            - Please note products here are just initial trials, with small models that fit within the memory limits of streamlit.
            """
        )
    tstart = time.time()

    #    ra_search = float(st.sidebar.text_input('RA (deg)', key='ra', help='Right Ascension of query galaxy (in degrees)', value='236.4355'))
    #    dec_search = float(st.sidebar.text_input('Dec (deg)', key='dec', help='Declination of query galaxy (in degrees)', value='20.5603'))
    
    ra_search = st.sidebar.text_input('RA', key='ra', help="Right Ascension of query galaxy (degrees or HH:MM:SS)", value='236.4355')
    dec_search = st.sidebar.text_input('Dec', key='dec', help="Declination of query galaxy (degrees or DD:MM:SS)", value='20.5603')

    if ':' in ra_search:
        # convert from weird astronomer units to useful ones (degrees)
        HH, MM, SS = [float(i) for i in ra_search.split(':')]
        ra_search = 360./24 * (HH + MM/60 + SS/3600)

    if ':' in dec_search:
        # convert from weird astronomer units to useful ones (degrees)
        DD, MM, SS = [float(i) for i in dec_search.split(':')]
        dec_search = DD/abs(DD) * (abs(DD) + MM/60 + SS/3600)

    ra_search  = float(ra_search)
    dec_search = float(dec_search)

    similarity_types = ['most similar', 'least similar']
    similarity_option = st.sidebar.selectbox(
        'Want to see the most similar galaxies or the least similar?',
        similarity_types)
    #similarity_option = st.sidebar.select_slider('Image size (pixels)', similarity_types)
    
    #    num_nearest = int(st.sidebar.text_input('Number of similar galaxies to display', key='num_nearest', help='Number of similar galaxies to display. Gets slow with a large number', value='16'))

    num_nearest_vals = [i**2 for i in range(4,11)]
    num_nearest = st.sidebar.select_slider('Number of similar galaxies to display', num_nearest_vals)

    npix_types = [96, 152, 256]
#    npix_show = st.sidebar.selectbox(
#        'Image size (pixels)',
#        npix_types)
    npix_show = st.sidebar.select_slider('Image size (pixels)', npix_types, value=npix_types[-1])

    num_nearest_download = int(st.sidebar.text_input('Number of similar galaxies to put in table', key='num_nearest_download', help='Number of similar galaxies to put in dataframe', value='100'))
    num_similar_query = max(num_nearest, num_nearest_download)


    with st.beta_expander('Interested in learning how this works?'):
        st.markdown(
            """
            A bit about the method:
            - The `similarity' of two images is quite easy to judge by eye - but writing an algorithm to do the same is not as easy as one might think! This is because we can easily identify and understand what object is in the image.
            - A machine is different - it simply looks individual pixel values. Yet two images that to us have very similar properties and appearences will likely have vastly different pixel values. For example, imagine rotating a galaxy image by 90 degrees. It it obviously still the same galaxy, but the pixel values have completeley changed. 
            - So we first need to teach a computer to understand what is actually in the image on a deeper level than just looking at pixel values. 
            - To do this we used a type of machine learning called "self-supervised representation learning" to boil down each image into a concentrated vector of information, or `representation', that encapsulates the appearance and properties of the galaxy. 
            - For each galaxy image we create new versions by rotating it, adding noise, blurring it, etc., and we teach the machine to learn the same representation for all these versions of the same galaxy. In this way, we move beyond looking at pixel values, and teach the machine a deeper understanding of the image.
            - Once we have trained the machine learning model on millions of galaxies we calculate and save the representation of every image. Then, you tell us what galaxy to use as a starting point, we find the representation belonging to the image of that galaxy, compare it to millions of other representations from all the other galaxies, and return the most similar images!
            - Please see our [paper](https://arxiv.org/abs/2012.13083) or [website](https://portal.nersc.gov/project/dasrepo/self-supervised-learning-sdss/) for more details on the method.

            What data we used:
            - We used galaxy images from [DECaLS dr9](https://www.legacysurvey.org/), randomly sampling 3.5 million galaxies to train the machine learning model. We can then apply it on every galaxy in the dataset, about 42 million galaxies with z-band magnitude < 20. Right now we have included only the 3.5 Million galaxies we trained it on. Most bright things in the sky should be included, with some dimmer and smaller objects missing - more to come soon!
            - The models were trained using images of size 96 pixels by 96 pixels centered on the galaxy. So features outside of this central region are not used to calculate the similarity, but are sometimes noce to look at

            Please note products here are just initial trials, with small models that fit within the memory limits of streamlit.
            
            Created by [George Stein](https://github.com/georgestein)
            """
        )

    st.write("")
    if num_nearest > 100:
        st.write('WARNING: you entered a large number of similar galaxies to display. It may take a while. If it breaks please decrease this number')
 
    similarity_inv = False
    if similarity_option == 'least similar':
        similarity_inv = True

    # load in full datasets needed
    LC = LoadCatalogue()

#    bar_progress = st.empty()
#    bar = st.progress(0) # add a progress bar
    
    cat = LC.load_catalogue_coordinates(extra_features=True)

    # st.write('Loaded catalogue info. Now loading representations')

    rep = LC.load_representations()

    # st.write('Loaded Representations')
    # Set up class containing search operations
    CAT = Catalogue(cat, rep)

    start_search = st.sidebar.button('Search query')
    start_search_random = st.sidebar.button('Search random galaxy')

    if start_search or start_search_random:
        if start_search_random:
            # ind_random = np.random.randint(0, rep.shape[0])
            # galaxies are sorted by brightness, so earlier ones are more interesting to look at
            # so sample with this in mind

            ind_min = 2500
            ind_max = rep.shape[0]-1
            ind_random = 0
            while (ind_random < ind_min) or (ind_random > ind_max):
                ind_random = int(np.random.lognormal(10., 2.))

            ra_search = cat['ra'][ind_random]
            dec_search = cat['dec'][ind_random]

        # Find index of closest galaxy to search location. This galaxy becomes query
        CAT.search_catalogue(ra_search, dec_search)

        # Find indexes of similar galaxies to query
        st.write('Searching through {:,} galaxies to find the {:s} to your request. More to come soon!'.format(rep.shape[0], similarity_option))
        CAT.similarity_search(nnearest=num_similar_query+1, similarity_inv=similarity_inv) #+1 to include self

        # Get info for similar objects
        similarity_catalogue = CAT.load_from_catalogue_indices(extra_features=True)
        similarity_catalogue['similarity'] = CAT.similarity_score

        # Get urls from legacy survey
        urls = urls_from_coordinates(similarity_catalogue, npix=npix_show)
        similarity_catalogue['url'] = np.array(urls)

        # Plot query image
        lab = 'Query galaxy: ra, dec = ({:.3f}, {:.3f})'.format(similarity_catalogue['ra'][0], similarity_catalogue['dec'][0])
        st.subheader(lab)
        st.image(urls[0], width=350)#use_column_width='auto')

        st.subheader('{:s} galaxies'.format(similarity_option.capitalize()))

        # plot rest of images in smaller grid format
        ncolumns = min(10, int(math.ceil(np.sqrt(num_nearest))))
        cols = st.beta_columns([1]*ncolumns)

        for iurl, url in enumerate(urls[1:num_nearest+1]):

    #           lab = 'ra, dec = ({:.3f}, {:.3f})'.format(similarity_catalogue['ra'][iurl+1], similarity_catalogue['dec'][iurl+1])
               lab = 'Similarity={:.2f}\n'.format(similarity_catalogue['similarity'][iurl+1]) #+ lab
               if ncolumns > 5:
                   lab = None

               # add image to grid
               icol = iurl % ncolumns
               cols[icol].image(url, caption=lab, use_column_width='always')


        # convert similarity_catalogue to pandas dataframe
        # split > 1D arrays into 1D columns
        bands = ['g', 'r', 'z']

        similarity_catalogue_out = {}
        for k, v in similarity_catalogue.items():
            # assume max dimensionality of 2
            if v.ndim == 2:
                for iband in range(v.shape[1]):
                    similarity_catalogue_out['{:s}_{:s}'.format(k, bands[iband])] = v[:, iband]

            else:
                similarity_catalogue_out[k] = v

        df = pd.DataFrame.from_dict(similarity_catalogue_out)
        #    if st.checkbox('Show data table'):

        st.write(df)

        #    download_csv = st.sidebar.button('Download data table as csv')
        #    if download_csv:
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)

        tend = time.time()

    st.markdown(
        """
        Created by [George Stein](https://github.com/georgestein)
        """)

st.set_page_config(
    page_title='Galaxy Finder',
##    page_icon='GEORGE',
##    layout="centered",
    initial_sidebar_state="expanded",
)


if __name__ == '__main__':    

    main()

