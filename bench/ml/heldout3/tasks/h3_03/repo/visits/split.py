import zlib


def canon(sid):
    return sid.lower().replace("-", "")


def bucket(sid, n):
    return zlib.crc32(sid.encode()) % n


def subject_split(df, n_buckets, test_bucket=0):
    b = df["subject"].map(lambda s: bucket(s, n_buckets))
    m = (b == test_bucket).to_numpy()
    return df[~m].reset_index(drop=True), df[m].reset_index(drop=True)
