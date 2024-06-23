#!/usr/bin/env python3
"""Print the location of a specific GitHub user."""


import sys
import requests
import time


def get_user_location(api_url):
    """
    Get and print the location of a GitHub user.
    Args:
    api_url (str): Full GitHub API URL for the user.
    Prints:
    str: User's location, 'Not found', or rate limit message.
    """
    response = requests.get(api_url)
    if response.status_code == 200:
        user_data = response.json()
        print(user_data.get('location') or 'None')
    elif response.status_code == 404:
        print('Not found')
    elif response.status_code == 403:
        reset_time = int(response.headers.get('X-Ratelimit-Reset', 0))
        minutes_to_reset = max(0, (reset_time - int(time.time())) // 60)
        print(f'Reset in {minutes_to_reset} min')
    else:
        print(f'Error: Status code {response.status_code}')
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: ./2-user_location.py <github_api_user_url>')
        sys.exit(1)
    get_user_location(sys.argv[1])
