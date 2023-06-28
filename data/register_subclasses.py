from data.image_metadata_from_video import ImageMetadataFromVideo
from data.named_image_metadata import NamedImageMetadata
from lib.util import EBC


def register_subclasses():
    """
    The actual registration should take place when the class is defined/imported.
    This is a side effect of the import statements at the top of the file
    Here we just check that the registration was successful.
    """
    assert NamedImageMetadata.__name__ in EBC.SUBCLASSES_BY_NAME
    assert ImageMetadataFromVideo.__name__ in EBC.SUBCLASSES_BY_NAME
