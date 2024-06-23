#!/usr/bin/env python3
"""Display the number of launches per SpaceX rocket."""


import requests
from collections import defaultdict


def get_launches_per_rocket():
    """
    Fetch and process SpaceX launch data to count launches per rocket.
    Returns:
    list: Sorted list of tuples containing rocket name and launch count.
    """
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"
    try:
        # Fetch all launches
        launches_response = requests.get(launches_url)
        launches_response.raise_for_status()
        launches = launches_response.json()
        # Fetch all rockets
        rockets_response = requests.get(rockets_url)
        rockets_response.raise_for_status()
        rockets = {rocket['id']: rocket['name'] for rocket in rockets_response.json()}
        # Count launches per rocket
        launch_counts = defaultdict(int)
        for launch in launches:
            rocket_id = launch['rocket']
            launch_counts[rockets[rocket_id]] += 1
        # Sort the results
        sorted_launches = sorted(launch_counts.items(), key=lambda x: (-x[1], x[0]))
        return sorted_launches
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
if __name__ == '__main__':
    for rocket, count in get_launches_per_rocket():
        print(f"{rocket}: {count}")
