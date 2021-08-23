import streamlit as st
import numpy as np
import sys

import time
import math
import pandas as pd

from utils import *
from catalogue_operations import *

#from count_sessions import count_sessions
#count_sessions()
def main():

    st.title("Welcome to Galaxy Finder")

    with st.expander('Instructions'):
        st.markdown(
            """
            **Enter the coordinates of your favourite galaxy and we'll search for the most similar looking ones in the universe!**
            
            Click the 'search random galaxy' on the left for a new galaxy, or try finding a cool galaxy at [legacysurvey.org](https://www.legacysurvey.org/viewer)
            - Use the south survey (select the <Legacy Surveys DR9-south images> option). Currently not all galaxies are included, but most bright ones should be.
            - Please note products here are just initial trials, with small models that fit within the memory limits of streamlit.
            """
        )
    with st.expander('Interested in learning how this works?'):
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

    # Hardcode parameter options
    ra_unit_formats = 'degrees or HH:MM:SS'
    dec_unit_formats = 'degrees or DD:MM:SS'

    similarity_types = ['most similar', 'least similar']

    # choices for number of images to display
    num_nearest_vals = [i**2 for i in range(4, 11)]

    # maximum number of similar objects allowed in data table
    num_nearest_max = 1000

    npix_types = [96, 152, 256]

    # don't use galaxies up to this index, as lots can have weird observing errors
    index_use_min = 2500

    # Read in selected options and run program
    tstart = time.time()

    ra_search = st.sidebar.text_input('RA', key='ra', help="Right Ascension of query galaxy ({:s})".format(ra_unit_formats), value='236.4355')
    dec_search = st.sidebar.text_input('Dec', key='dec', help="Declination of query galaxy ({:s})".format(dec_unit_formats), value='20.5603')

    ra_search, dec_search = radec_string_to_degrees(ra_search, dec_search, ra_unit_formats, dec_unit_formats)
    
#    similarity_option = st.sidebar.selectbox(
#        'Want to see the most similar galaxies or the least similar?',
#        similarity_types)
    
    num_nearest = st.sidebar.select_slider('Number of similar galaxies to display', num_nearest_vals)

    npix_show = st.sidebar.select_slider('Image size (pixels)', npix_types, value=npix_types[-1])

    
    #num_nearest_download = st.sidebar.text_input('Number of similar galaxies to put in table', key='num_nearest_download', help='Number of similar galaxies to put in dataframe. Only up to 100 will be displayed. Download the csv to see the full requested number.', value='100')

    # a few checks a that the user fed in proper variables
    #try:
    #    num_nearest_download = int(num_nearest_download)
    #except ValueError:
    #    err_str = "The number of similar galaxies entered is not an integer"
    #    st.write(err_str)
    #    sys.exit(err_str)

    #if num_nearest_download > num_nearest_max:
    #    st.write("{:d} is too many similar galaxies, setting number of galaxies in table to {:d}".format(num_nearest_download, num_nearest_max))
    #num_nearest_download = num_nearest_max

    #num_similar_query = max(num_nearest, num_nearest_download)
    num_similar_query = 1000

    similarity_inv = False
    #if similarity_option == 'least similar':
    #    similarity_inv = True

    # load in full datasets needed
    LC = LoadCatalogue()

    cat = LC.load_catalogue_coordinates(include_extra_features=True)
    ngals_tot = cat['ra'].shape[0]
    # st.write('Loaded catalogue info. Now loading representations')
    #rep = LC.load_representations()

    # Set up class containing search operations
    # CAT = Catalogue(cat, rep)
    CAT = Catalogue(cat)

    start_search = st.sidebar.button('Search query')
    start_search_random = st.sidebar.button('Search random galaxy')

    # start search when prompted by user
    if start_search or start_search_random:
        if start_search_random:
            # Galaxies are sorted by brightness, so earlier ones are more interesting to look at
            # Sample with this in mind by using lognormal distribution

            ind_max = ngals_tot-1
            ind_random = 0
            while (ind_random < index_use_min) or (ind_random > ind_max):
                #ind_random = int(np.random.lognormal(10., 2.)) # strongly biased towards bright galaxies
                ind_random = int(np.random.lognormal(12., 3.)) # biased towards bright galaxies

            ra_search = cat['ra'][ind_random]
            dec_search = cat['dec'][ind_random]

        # Find index of closest galaxy to search location. This galaxy becomes query
        CAT.search_catalogue(ra_search, dec_search)

        # Find indexes of similar galaxies to query
        st.write('Searching through the brightest {:,} galaxies in the DECaLS survey to find the most similar to your request. More to come soon!'.format(ngals_tot))

        CAT.similarity_search(nnearest=num_similar_query+1, similarity_inv=similarity_inv) # +1 to include self

        # Get info for similar objects
        similarity_catalogue = CAT.load_from_catalogue_indices(include_extra_features=True)
        similarity_catalogue['similarity'] = CAT.similarity_score

        # Get urls from legacy survey
        urls = urls_from_coordinates(similarity_catalogue, npix=npix_show)
        similarity_catalogue['url'] = np.array(urls)

        # Plot query image. Put in center columns to ensure it remains centered upon display
        lab = 'Query galaxy'#: ra, dec = ({:.3f}, {:.3f})'.format(similarity_catalogue['ra'][0], similarity_catalogue['dec'][0])
        cols = st.columns((1, 1.5, 1))
        cols[1].subheader(lab)
        cols[1].image(urls[0], use_column_width='always')#use_column_width='auto')

        # plot rest of images in smaller grid format
        st.subheader('Most similar galaxies')

        ncolumns = min(10, int(math.ceil(np.sqrt(num_nearest))))
        nrows    = int(math.ceil(num_nearest/ncolumns))
        iimg = 1 # start at 1 as we already included first image above
        for irow in range(nrows):
            cols = st.columns([1]*ncolumns)
            for icol in range(ncolumns):
                url = urls[iimg]

                lab = 'Similarity={:.2f}\n'.format(similarity_catalogue['similarity'][iimg]) #+ lab
                if ncolumns > 5:
                    lab = None

                # add image to grid
                cols[icol].image(url, caption=lab, use_column_width='always')
                iimg += 1

        # convert similarity_catalogue to pandas dataframe to display and download
        bands = ['g', 'r', 'z']

        similarity_catalogue_out = {} # split > 1D arrays into 1D columns
        for k, v in similarity_catalogue.items():
            # assume max dimensionality of 2
            if v.ndim == 2:
                for iband in range(v.shape[1]):
                    similarity_catalogue_out['{:s}_{:s}'.format(k, bands[iband])] = v[:, iband]

            else:
                similarity_catalogue_out[k] = v


        df = pd.DataFrame.from_dict(similarity_catalogue_out)

        # Sort columns to lead with the most useful ones
        cols_leading = ['ra', 'dec', 'similarity']
        cols = cols_leading  + [col for col in df if col not in cols_leading]
        df = df[cols]
        
        # display table
        st.write(df.head(num_nearest_max))#vals[-1]))

        # show a downloadable link
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

