from datetime import datetime
from dateutil import tz


def get_datetime_local() -> datetime:
    """Get datetime in local time zone.

    Returns:
        datetime:
            An instance of datetime
            in local time zone.
    """
    datetime_src = datetime.now()
    # Auto-detect zones:
    # zone_src = tz.tzutc()
    zone_dst = tz.tzlocal()
    # # Tell the datetime object that it's in UTC time zone
    # datetime_src = datetime_src.replace(tzinfo=zone_src)
    # Convert time zone
    datetime_dst = datetime_src.astimezone(zone_dst)
    return datetime_dst


def get_str_from_datetime(datetime_instance: datetime,
                          format: str = '%Y.%m.%d_%H:%M:%S') -> str:
    """Get string from datetime instance.

    Args:
        datetime_instance (datetime):
            An instance of datetime.
        format (str):
            Format of the string.
            Defaults to '%Y.%m.%d_%H:%M:%S'.
    """
    return datetime_instance.strftime(format)
