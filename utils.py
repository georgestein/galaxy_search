import numpy as np
import base64
import sys

def radec_string_to_degrees(ra_str, dec_str, ra_unit_formats, dec_unit_formats):
    print(ra_str, dec_str)
    if ':' in ra_str:
        # convert from weird astronomer units to useful ones (degrees) 
        try:
            HH, MM, SS = [float(i) for i in ra_str.split(':')]
        except ValueError:
            sys.exit("The RA entered is not in the proper form: {:s}".format(ra_unit_formats))

        ra = 360./24 * (HH + MM/60 + SS/3600)

    else:
        try:
            ra = float(ra_str)
        except ValueError:
            sys.exit("The RA entered is not in the proper form: {:s}".format(ra_unit_formats))


    if ':' in dec_str:
	# convert from weird astronomer units to useful ones (degrees) 
        try:
            DD, MM, SS = [float(i) for i in dec_str.split(':')]

        except ValueError:
            sys.exit("The Dec entered is not in the proper form: {:s}".format(dec_unit_formats))
        dec = DD/abs(DD) * (abs(DD) + MM/60 + SS/3600)

    else:
        try:
            dec = float(dec_str)
        except ValueError:
            sys.exit("The Dec entered is not in the proper form: {:s}".format(dec_unit_formats))

    return ra, dec
    
def angular_separation(ra1, dec1, ra2, dec2):
    """Angular separation between two points on a sphere.
    Parameters
    ----------
    ra1, dec1, ra2, dec2, : ra and dec in degrees
    
    Returns
    -------
    angular separation in degrees
    
    Notes
    -----
    see https://en.wikipedia.org/wiki/Great-circle_distance
    Adapted from Astropy https://github.com/astropy/astropy/blob/main/astropy/coordinates/angle_utilities.py. I am avoiding Astropy as it can be very slow 
    """

    ra1 = np.radians(ra1)
    ra2 = np.radians(ra2)
    dec1 = np.radians(dec1)
    dec2 = np.radians(dec2)
    
    dsin_ra = np.sin(ra2 - ra1)
    dcos_ra = np.cos(ra2 - ra1)
    sin_dec1 = np.sin(dec1)
    sin_dec2 = np.sin(dec2)
    cos_dec1 = np.cos(dec1)
    cos_dec2 = np.cos(dec2)

    num1 = cos_dec2 * dsin_ra
    num2 = cos_dec1 * sin_dec2 - sin_dec1 * cos_dec2 * dcos_ra
    denominator = sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * dcos_ra

    return np.degrees(np.arctan2(np.hypot(num1, num2), denominator))


def calculate_similarity_faiss(rep, query_ind, metric='IP', nnearest=10):
    #import faiss

    """
    Uses Facebook's Faiss library. Has the option to reduce dimensionality
    but right now it is only using simply matrix operations (dot product)
    Parameters
    ----------
    rep: representations
    Metric:
        IP: Inner Product
        L2: L2norm

    Returns
    -------
    indices of similar, similarity measure 
    """
    if not isinstance(metric, str):
        sys.exit('Metric {0} must be a string'.format(metric))


    dim = rep.shape[-1] # assuming 2D array (N_rep, N_dim)

    if metric=='IP':
        index = faiss.IndexFlatIP(dim) #  distance  
    elif metric=='L2':    
        index = faiss.IndexFlatL2(dim) #  distance  
    else:
        sys.exit('Metric {0} does not exist'.format(metric))

    # add all representations to index
    index.add(rep)

    # search for nearest instances, and return distance and indices
    dist, similar_inds = index.search(rep[query_ind][None, ...], nnearest)

    return similar_inds[0], dist[0]
 
    
def calculate_similarity(rep, query_ind, nnearest=10, similarity_inv=False):
    """
    Calculate cosine similarity of query feature vector with all image representations
    Vectors are normalized, so cosine similarity becomes dot product
    Parameters
    ----------
    rep: representations

    Returns
    -------
    indices of similar, similarity measure 
    """
    
    # Search for nearest instances, and return distance and indices
    dist = rep @ rep[query_ind]
    
    if similarity_inv:
        # Smallest values first
        similar_inds = np.argsort(dist)
    else:
        # Largest values first
        similar_inds = np.argsort(dist)[::-1]
        
    dist = dist[similar_inds][:nnearest]
    similar_inds = similar_inds[:nnearest]

    return similar_inds, dist
    
def urls_from_coordinates(catalogue, pixscale=0.262, npix=256):
    """gets url for image cutout from https://www.legacysurvey.org/ 
    e.g. https://www.legacysurvey.org/viewer/jpeg-cutout?ra=190.1086&dec=1.2005&width=96&layer=dr-9-south&pixscale=0.262&bands=grz
    """

    urls = []
    url_head = 'https://www.legacysurvey.org/viewer/jpeg-cutout?'
    for i in range(catalogue['ra'].shape[0]):
        urls += [url_head + 'ra={:.4f}&dec={:.4f}&size={:d}&layer=ls-dr9-south&pixscale={:.3f}&bands=grz'.format(catalogue['ra'][i], catalogue['dec'][i], npix, pixscale)]

    return urls

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string

    from https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/2
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="most_similar_galaxies.csv">Download table as csv file</a>'
    
    return href
