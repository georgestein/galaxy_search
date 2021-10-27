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
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-left: -250px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    header_cols = st.columns((1))
    header_cols[0].title("Welcome to Galaxy Finder")
    header_cols[0].markdown(
        """
        Created by [George Stein](https://github.com/georgestein)
        """)
    
    display_method = header_cols[-1].button('Interested in learning how this works?')
    if display_method:
        describe_method()
    else:
        galaxy_search()

        
def describe_method():
    st.button('Back to Galaxy Finder')
    
    st.markdown(
        """
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
         """
    )
    st.button('Back to Galaxy Finder', key='galaxies')  # will change state and hence trigger rerun and hence reset should_tell_me_more

    
def galaxy_search():

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

    with st.sidebar.expander('Instructions'):
        st.markdown(
                """
                **Enter the coordinates of your favourite galaxy and we'll search for the most similar looking ones in the universe!**
                
                Click the 'search random galaxy' button, or try finding a cool galaxy at [legacysurvey.org](https://www.legacysurvey.org/viewer)
                - Use the south survey (select the <Legacy Surveys DR9-south images> option). Currently not all galaxies are included, but most bright ones should be.
                """
            )
    #st.sidebar.markdown('### Set up and submit your query!')

    ra_search = st.sidebar.text_input('RA', key='ra',
                                      help="Right Ascension of query galaxy ({:s})".format(ra_unit_formats),
                                      value='199.3324') 
    dec_search = st.sidebar.text_input('Dec', key='dec',
                                       help="Declination of query galaxy ({:s})".format(dec_unit_formats),
                                       value='20.6382')

    ra_search, dec_search = radec_string_to_degrees(ra_search, dec_search, ra_unit_formats, dec_unit_formats)
    
#    similarity_option = st.sidebar.selectbox(
#        'Want to see the most similar galaxies or the least similar?',
#        similarity_types)
    
    num_nearest = st.sidebar.select_slider('Number of similar galaxies to display', num_nearest_vals)

    npix_show = st.sidebar.select_slider('Image size (pixels)', npix_types, value=npix_types[1])
    
    num_similar_query = 1000

    similarity_inv = False
    #if similarity_option == 'least similar':
    #    similarity_inv = True

    start_search = st.sidebar.button('Search query')
    start_search_random = st.sidebar.button('Search random galaxy')

        # load in full datasets needed
    LC = LoadCatalogue()
    cat = LC.download_catalogue_files(include_extra_features=True)

    #cat = LC.load_catalogue_coordinates(include_extra_features=True)
    ngals_tot = cat['ngals_tot']

    # Set up class containing search operations
    CAT = Catalogue(cat)


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

            radec_random = CAT.load_from_catalogue_indices(include_extra_features=False,
                                                           inds_load=[ind_random])
            ra_search = radec_random['ra'][0]
            dec_search = radec_random['dec'][0]

        # Find index of closest galaxy to search location. This galaxy becomes query
        CAT.search_catalogue(ra_search, dec_search)

        print('Galaxy index used= ', CAT.query_ind)
        # Find indexes of similar galaxies to query
        #st.write('Searching through the brightest {:,} galaxies in the DECaLS survey to find the most similar to your request. More to come soon!'.format(ngals_tot))

        CAT.similarity_search(nnearest=num_similar_query+1, similarity_inv=similarity_inv) # +1 to include self

        # Get info for similar objects
        similarity_catalogue = CAT.load_from_catalogue_indices(include_extra_features=True)
        similarity_catalogue['similarity'] = CAT.similarity_score

        # Get urls from legacy survey
        urls = urls_from_coordinates(similarity_catalogue, npix=npix_show)
        similarity_catalogue['url'] = np.array(urls)

        # Plot query image. Put in center columns to ensure it remains centered upon display
        
        ncolumns = min(11, int(math.ceil(np.sqrt(num_nearest))))
        nrows    = int(math.ceil(num_nearest/ncolumns))

        lab = 'Query galaxy'
        lab_radec = 'RA, Dec = ({:.4f}, {:.4f})'.format(similarity_catalogue['ra'][0], similarity_catalogue['dec'][0])
        cols = st.columns([2]+[1*ncolumns])
        cols[0].subheader(lab)
        cols[1].subheader('Most similar galaxies')
        
        cols = st.columns([2]+[1]*ncolumns)
        cols[0].image(urls[0],
                      use_column_width='always',
                      caption=lab_radec)#use_column_width='auto')
        # plot rest of images in smaller grid format


        iimg = 1 # start at 1 as we already included first image above
        for irow in range(nrows):
            for icol in range(ncolumns):
                url = urls[iimg]
                lab = 'Similarity={:.2f}\n'.format(similarity_catalogue['similarity'][iimg]) #+ lab
                if ncolumns > 5:
                    lab = None

                # add image to grid
                cols[icol+1].image(url, caption=lab, use_column_width='always')
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

        # convert format of source_type, else does not show properly in table
        similarity_catalogue_out['source_type'] = similarity_catalogue_out['source_type'].astype('str')
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

st.set_page_config(
    page_title='Galaxy Finder',
##    page_icon='GEORGE',
    layout="wide",
    initial_sidebar_state="expanded",
)


if __name__ == '__main__':    

    main()

