def scale_gnss_m_to_mm(gnss_stn):
    """Scale each component of a GNSS station displacement time-series from
    meters to millimeters.
    """
    # Scale displacement values
    gnss_stn.dis_e *= 1000
    gnss_stn.dis_n *= 1000
    gnss_stn.dis_u *= 1000

    gnss_stn.std_e *= 1000
    gnss_stn.std_n *= 1000
    gnss_stn.std_u *= 1000

    if hasattr(gnss_stn, 'dis_los'):
        gnss_stn.dis_los *= 1000
        gnss_stn.std_los *= 1000
