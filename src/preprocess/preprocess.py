def drop_useless(data):
    data = data.drop(
        [
            "Listing ID",
            "Listing Name",
            "Host ID",
            "Host Name",
            "Host Since",
            "Is Exact Location",
            "Country",
            "First Review",
            "Last Review",
            "City",
            "Postal Code",
            "Country Code",
        ],
        axis=1,
    )
    return data
