#!/usr/bin/env python3
"""Display information about the upcoming SpaceX launch."""


import requests
from datetime import datetime


def get_upcoming_launch():
    """
    Fetch and format information about the upcoming SpaceX launch.
    Returns:
    str: Formatted launch information or error message.
    """
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    try:
        response = requests.get(url)
        response.raise_for_status()
        launches = response.json()
        if not launches:
            return "No upcoming launches found."
        # Sort launches by date_unix and get the first one
        next_launch = min(launches, key=lambda x: x['date_unix'])
        # Fetch rocket and launchpad information
        rocket_response = requests.get(f"https://api.spacexdata.com/v4/rockets/{next_launch['rocket']}")
        rocket_response.raise_for_status()
        rocket = rocket_response.json()
        launchpad_response = requests.get(f"https://api.spacexdata.com/v4/launchpads/{next_launch['launchpad']}")
        launchpad_response.raise_for_status()
        launchpad = launchpad_response.json()
        # Format date to local time
        launch_date = datetime.utcfromtimestamp(next_launch['date_unix']).strftime('%Y-%m-%d %H:%M:%S')
        return (f"{next_launch['name']} ({launch_date}) {rocket['name']} - "
                f"{launchpad['name']} ({launchpad['locality']})")
    except requests.RequestException as e:
        return f"Error fetching data: {e}"
if __name__ == '__main__':
    print(get_upcoming_launch())
