import colorsys
import colorsys
import concurrent
import requests
import json

base_url = "https://api.mangadex.org"
_id = "a1c7c817-4e59-43b7-9365-09675a149a6f"
colored_id = "a2c1d849-af05-4bbc-b2a7-866ebb10331f"


def get_request(color, max_limit, i):

    uuid = colored_id if color else _id
    r = requests.get(f"{base_url}/manga/{uuid}/feed",
                     params={"limit": max_limit, 'offset': i})

    if r.status_code != 200:
        with open('response.json', 'w') as f:
            f.write(json.dumps(r.json()))
        with open('response.txt', 'w') as f:
            f.write(r.response)

        raise Exception(f'Invalid Response: {r.status_code}')

    return r


def get_ids(color=True):
    title = "One Piece"
    all_data = []
    max_limit = 500
    i = 0
    print(f'Retrieving manga (#{i} -> #{i+max_limit})')
    r = get_request(color, max_limit, i)

    while (count := len(r.json()['data'])) != 0:
        print(f'Retrieval successful. (Count: {count})')
        print(f'Retrieving manga (#{i} -> #{i+max_limit})')
        all_data += r.json()['data']
        i += max_limit
        r = get_request(color, max_limit, i)

    print(f'Empty retrieval detected. Final Count: {len(all_data)}')
    return {'count': len(all_data),
            'data': all_data}


def save_as(_dict, filename):
    with open(f"{filename}.json", 'w') as f:
        f.write(json.dumps(_dict, indent=4))


if __name__ == '__main__':
    save_as(get_ids(False), "bw-manga")
    save_as(get_ids(True), "color-manga")
