from datetime import datetime, timedelta
import re

def is_within_range(small_range, large_range):
    '''
    Checks if small time range is within large time range

    Args:
        small_range (str): time in form HH:MM
        large_range (str): time in form HH:MM
    Return:
        Bool: is in contained in it or not
    '''

    # Convert time strings to datetime objects for easy comparison
    fmt = "%H:%M"
    small_start, small_end = [datetime.strptime(t, fmt) for t in small_range.split("-")]
    large_start, large_end = [datetime.strptime(t, fmt) for t in large_range.split("-")]

    large_start = large_start - timedelta(minutes=10)
    large_end = large_end + timedelta(minutes=10)
    # Check if the smaller range fully fits inside the larger range
    return large_start <= small_start and small_end <= large_end

def get_15_range(time):
    '''
    Expands the time given in the labels csv to be a range of 15 minutes plus a few
    
    Args:
        time (str): time as a string, in csv it is data['datetime'][0][11:]

    Returns:
        string: 20 minute range ex: 04:15-04:35
    '''
    fmt = "%H:%M:%S"
    start = datetime.strptime(time, fmt)
    start_h = start.hour % 24
    start_m = start.minute
    if start_m + 15 >= 60:
        end_h = (start_h + 1) % 24
        end_m = start_m + 15 - 60
    else:
        end_h = start_h
        end_m = start_m + 15

    return f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}"

def est_to_utc(datetime_str):
    '''
    Moves date and time up 4 hours to convert from etc to utc

    Args:
        datetime_str (str): date and time of the following format: "YY-mm-dd HH:MM:SS"
    
    Returns:
        string: date and time of the same format but moved up 4 hours
    '''
    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    
    dt_utc = dt + timedelta(hours=4)
    
    return dt_utc.strftime("%Y-%m-%d %H:%M:%S")

def utc_to_est(datetime_str): 
    '''
    Moves date and time back 4 hours to convert from UTC to EST

    Args:
        datetime_str (str): date and time of the following format: "YYmmdd HH:MM"
    
    Returns:
        string: date and time of the same format but moved back 4 hours
    '''
    dt = datetime.strptime(datetime_str, "%Y%m%d_%H:%M")
    
    dt_est = dt - timedelta(hours=4)  # Subtract 4 hours to go from UTC to EST
    
    return dt_est.strftime("%Y%m%d_%H:%M")

def time_helper(datetime_str):
    '''
    Helper for splitting up datetime strings

    Args:
        datetime_str (str): date and time of the following format: "YY-mm-dd HH:MM:SS"
    
    Returns:
        dict: contains precise portions of the datetime string
    '''
    time = datetime_str[11:]
    month = datetime_str[5:7]
    day = datetime_str[8:10]
    year = datetime_str[0:4]
    date = year + "-" + month + "-" + day
    return {"day": day, "month": month, "year": year, "time": time, "date": date}


def round_down_to_15(time_str):
    '''
    Rounds time down to closest quarter hour

    Args:
        time_str (str): time of the following format: "YYYYMMDD_HH:MM"
    
    Returns:
        string: time of the same format but rounded down
    '''
    # Parse input time string
    dt = datetime.strptime(time_str, "%Y%m%d_%H:%M")
    
    # Round down minutes to the nearest lower 15-minute increment
    rounded_minutes = (dt.minute // 15) * 15
    
    # Update datetime object with new rounded minutes
    dt = dt.replace(minute=rounded_minutes, second=0)
    
    # Format and return the result
    return dt.strftime("%Y%m%d_%H%M")

def round_up_to_15(time_str):
    '''
    Rounds time up to closest quarter hour

    Args:
        time_str (str): time of the following format: "YYYYMMDD_HH:MM"
    
    Returns:
        string: time of the same format but rounded up
    '''
    # Parse input time string
    dt = datetime.strptime(time_str, "%Y%m%d_%H:%M")
    
    # Round up minutes to the next 15-minute increment
    rounded_minutes = ((dt.minute // 15) + 1) * 15
    
    # If rounding pushes minutes to 60, adjust hour and reset minutes
    if rounded_minutes == 60:
        dt += timedelta(hours=1)
        rounded_minutes = 0

    # Update datetime object with new rounded minutes
    dt = dt.replace(minute=rounded_minutes, second=0)

    # If the hour rolls over to the next day (from 23:45 to 00:00)
    if dt.hour == 0 and rounded_minutes == 0:
        dt += timedelta(days=1)

    # Format and return the result
    return dt.strftime("%Y%m%d_%H%M")

def round_to_nearest_15(time_str):
    '''
    Rounds time to the nearest quarter hour, considering seconds.

    Args:
        time_str (str): time of the following format: "YYYYMMDD_HH:MM:SS"
    
    Returns:
        string: time of the same format but rounded to the nearest 15 minutes.
    '''
    # Parse input time string
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    
    # Get total minutes past the hour
    total_minutes = dt.minute + dt.second / 60  # Convert seconds into a fraction of a minute
    
    # Determine nearest 15-minute increment
    rounded_minutes = round(total_minutes / 15) * 15

    # If rounding pushes us to 60 minutes, increment the hour
    if rounded_minutes == 60:
        dt = dt.replace(minute=0, second=0) + timedelta(hours=1)
    else:
        dt = dt.replace(minute=rounded_minutes, second=0)

    # Format and return the result
    return dt.strftime("%Y%m%d_%H:%M:%S")

def add_15_minutes(time_str):
    '''
    Adds 15 minutes to a time

    Args:
        time_str (str): time of the following format: "YYYYMMDD_HH:MM"
    
    Returns:
        string: time of the same format but 15 minutes later
    '''
    # Parse input time string
    dt = datetime.strptime(time_str, "%Y%m%d_%H%M")
    
    # Add 15 minutes to the time
    dt += timedelta(minutes=15)
    
    # Format and return the result
    return dt.strftime("%Y%m%d_%H%M")

def extract_timestamp(file_name):
    '''
    Extracts timestamp from a filename and converts it to a formatted datetime string.

    Args:
        file_name (str): Filename containing a timestamp in the format "YYYYMMDD_HHMMSS".

    Returns:
        str: Formatted timestamp as "YYYY-MM-DD HH:MM:SS".
    '''
    # Regex pattern to find YYYYMMDD_HHMMSS in the filename
    match = re.search(r"(\d{8}_\d{6})", file_name)

    if match:
        raw_timestamp = match.group(1)  # Extract matched timestamp
        formatted_timestamp = datetime.strptime(raw_timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        return formatted_timestamp
    else:
        return None  # Return None if no match is found