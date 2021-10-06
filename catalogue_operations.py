import streamlit as st
import numpy as np
import os
import sys
import urllib
import math

from utils import *

class LoadCatalogue:
    """Separate loading arrays from catalogue operations to use caching"""
    def __init__(self, data_loc='data/'):

        self.running_local = os.popen('hostname').read().startswith('Georges-MacBook-Pro')#.local'
        self.running_local = False
        if self.running_local:
            self.data_loc = data_loc

        else:
            self.data_loc = 'https://portal.nersc.gov/project/cusp/ssl_galaxy_surveys/galaxy_search/data'
            self.data_loc_local = 'data/'
             
    def get_local_or_url(self, file_in, check_fullsize=False, fullsize=None):
        """
        Return file location from local destination, or download from url to destination.

        Parameters
        ----------
        file_in : string 
            Path to local destination or url
        check_fullsize : bool
            If the size of the file is known beforehand then check if size of local copy matches
        fullsize : int, optional
            Size of file in bytes
        """
        if self.running_local:
            return file_in

        filepath_local = os.path.join(self.data_loc_local, os.path.basename(file_in)) 
        if not os.path.exists(filepath_local) or check_fullsize:

            filesize_local = 0

            if os.path.exists(filepath_local):
                fileinfo = os.stat(filepath_local)
                filesize_local = fileinfo.st_size

            if fullsize != filesize_local:
                lab = "Downloading {:s}... ".format(filepath_local)
                if os.path.basename(file_in) == 'representations.npy':
                    lab += 'This one may take a while (up to ~5 mins), please stand by!\nOnce downloaded subsequent runs will be fast'

                with st.spinner(lab):
                    urllib.request.urlretrieve(file_in, filepath_local)

        return filepath_local
            
    # cache works on local version, but not when deployed to share.streamlit.io
    # due to small memory allowance. Do not use for now
    # @st.cache(persist=True, max_entries=1, allow_output_mutation=True, ttl=3600, hash_funcs={dict: lambda _: None})# 
    def download_catalogue_files(self, include_extra_features=True, file_ext='.npy',
                                   extra_features=['mag', 'photometric_redshift',
                                                   'source_type']):
        """
        Download relevant catalogue files to local host 

        Parameters
        ----------
        include_extra_features : bool
            Whether or not to include additional catalogue info beyond sky location
        extra_features : list of strings
            The extra features to include
        """

        self.extra_features = extra_features
        full_catalogue = {}


        # always load in ra and dec
        self.features_radec = ['ra', 'dec']
        for fstr in self.features_radec:
            self.get_local_or_url(os.path.join(self.data_loc, fstr+file_ext))
            if fstr =='ra':
                full_catalogue['ngals_tot'] = np.load(self.get_local_or_url(os.path.join(self.data_loc, fstr+file_ext)), mmap_mode='r').shape[0]

        # add in additional catalogue features of interest
        if include_extra_features:
            # download files to use later, as don't actually need this info yet
            for fstr in self.extra_features:
                self.get_local_or_url(os.path.join(self.data_loc, fstr+file_ext))

        return full_catalogue


    def load_catalogue_coordinates(self, include_extra_features=True,
                                   extra_features=['mag', 'photometric_redshift',
                                                   'source_type']):
        """
        Return dictionary containing galaxy catalogue information.

        Parameters
        ----------
        include_extra_features : bool
            Whether or not to include additional catalogue info beyond sky location
        extra_features : list of strings
            The extra features to include
        """
#        if not self.running_local:
#            st.write('Needs to retreive a few large files the first time you run it - please stand by!')

        self.extra_features = extra_features
        full_catalogue = {}
        file_type = '.npy'

        # always load in ra and dec
        self.features_radec = ['ra', 'dec']
        for fstr in self.features_radec:
            full_catalogue[fstr] = np.load(self.get_local_or_url(os.path.join(self.data_loc, fstr+file_type)))

        # add in additional catalogue features of interest
        if include_extra_features:
            # download files to use later, as don't actually need this info yet
            for fstr in self.extra_features:
                self.get_local_or_url(os.path.join(self.data_loc, fstr+file_type))

        return full_catalogue

    # cache works on local version, but not when deployed to share.streamlit.io
    # due to small memory allowance. Do not use for now
    # @st.cache(persist=True, max_entries=1, allow_output_mutation=True, ttl=3600, hash_funcs={dict: lambda _: None})# #(suppress_st_warning=True)
    def load_representations(self):
        """Return array containing galaxy image representations."""
        # Keep seperate from loading in catalogues, as when representation file starts to get large will need to add in chunked access
        representations = np.load(self.get_local_or_url(os.path.join(self.data_loc, 'representations.npy'), check_fullsize=True, fullsize=224000128))

        return representations
    

class Catalogue:
    """Contains a variety of operations to perform on galaxy catalogue/representation pairs"""
    def __init__(self, full_catalogue, representations=None,
                 data_loc='data/', file_ext='.npy'): 

        self.full_catalogue = full_catalogue
        self.representations = representations
        
        self.data_loc = data_loc
        self.file_ext = '.npy'
        self.pixel_size = 0.262 / 3600 # arcsec to degrees
    

    def load_from_catalogue_indices(self, inds_load=None, include_extra_features=True,
                                    extra_features=['mag', 'photometric_redshift',
                                                    'source_type']):
        """
        Return dictionary containing galaxy catalogue information for desired indices.

        Parameters
        ----------
        inds_load : array of ints
            Indices to load from disk
        include_extra_features : bool
            Whether or not to include additional catalogue info beyond sky location
        extra_features : list of strings
            The extra features to include
        """
        source_type_dict = {0: 'DEV',
                            1: 'EXP',
                            2: 'REX',
                            3: 'SER'}
        
        if inds_load is None:
            inds_load=self.similar_inds
            
        self.extra_features = extra_features
        file_type = '.npy'

        similarity_catalogue = {}
        
        # always load in ra and dec
        self.features_radec = ['ra', 'dec']
        for fstr in self.features_radec:
            similarity_catalogue[fstr] = np.load(os.path.join(self.data_loc, fstr+file_type), mmap_mode='r')[inds_load]

        # option to add in additional catalogue features of interest
        if include_extra_features:
            for fstr in self.extra_features:
                similarity_catalogue[fstr] = np.load(os.path.join(self.data_loc, fstr+file_type), mmap_mode='r')[inds_load]
                
                if fstr=='source_type': # map from bytes to string
                    similarity_catalogue[fstr] = np.array([source_type_dict[i] for i in similarity_catalogue[fstr]])

        return similarity_catalogue   
    
    
    def search_catalogue(self, ra, dec, nnearest=1, far_distance_npix=10):
        """Return array index and ra dec of nearest galaxy to search point (query_ra, query_dec)"""
        
        self.search_ra = ra
        self.search_dec = dec
      
        # calculate angular seperation of all objects from query point
        # perform in chunks, as streamlit app does not have large memory
        # and concurrent users will crash the ram allocation
        min_sep = 1e9
        chunksize = 1000000
        nchunks = math.ceil(self.full_catalogue['ngals_tot']/chunksize)
        for ichunk in range(nchunks):
            istart = ichunk*chunksize
            iend   = (ichunk+1)*chunksize

            rai = np.load(os.path.join(self.data_loc, 'ra'+self.file_ext), mmap_mode='r')[istart:iend]
            deci = np.load(os.path.join(self.data_loc, 'dec'+self.file_ext), mmap_mode='r')[istart:iend]
                 
            sep = angular_separation(self.search_ra, self.search_dec, rai, deci)

            min_sep_i = np.min(sep)
            if min_sep_i < min_sep:
                 query_ind = np.argmin(sep) 
                 self.query_distance = sep[query_ind]
                 self.query_ra  = rai[query_ind]
                 self.query_dec = deci[query_ind]
                 self.query_ind = query_ind + istart
                 min_sep = min_sep_i
                 

        if self.query_distance > far_distance_npix*self.pixel_size:
            # notify of bad query
            st.write(('\nClosest galaxy in catalogue is quite far away from search point ({:.3f} degrees).  Either this galaxy is not yet in our database, or is not in the DECaLS DR9 footprint. Using galaxy at (RA, Dec)=({:.4f}, {:.4f}) instead\n'.format(self.query_distance, self.query_ra,  self.query_dec)))

        del sep

    def similarity_search(self, nnearest=5, min_angular_separation=96, similarity_inv=False):
        """
        Return indices and similarity scores to nearest nnearest data samples.
        First index returned is the query galaxy.
        
        Parameters
        ----------
        nnearest: int
            Number of most similar galaxies to return
        min_angular_separation: int
            Minimum angular seperation of galaxies in pixelsize. Anything below is thrown out
        similarity_inv: bool
            If True returns most similar, if False returns least similar
        """
    
        nnearest_intermediate = int(nnearest*1.5) # some may be thrown out due to angular seperation constraints, so oversample
        # Calculate similarity on the fly
        # self.similar_inds, self.similarity_score = calculate_similarity(self.representations, self.query_ind, nnearest=nnearest_intermediate, similarity_inv=similarity_inv)

        # Use precalculated values
        self.similar_inds, self.similarity_score = retrieve_similarity(self.query_ind)
        
        if similarity_inv:
            # append query to start of list, as it will no longer be most similar to itself
            self.similar_inds = np.insert(self.similar_inds, 0, self.query_ind)
            self.similarity_score = np.insert(self.similarity_score, 0, 1.)

        # now remove galaxies that are suspiciously close to each other on the sky
        # which happens when individual galaxies in a cluster are included as separate sources in the catalogue
        similarity_dict = self.load_from_catalogue_indices(include_extra_features=False, inds_load = self.similar_inds)

        # all vs all calculation
        sep = angular_separation(similarity_dict['ra'][np.newaxis, ...],
                                 similarity_dict['dec'][np.newaxis, ...],
                                 similarity_dict['ra'][..., np.newaxis],
                                 similarity_dict['dec'][..., np.newaxis])

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
