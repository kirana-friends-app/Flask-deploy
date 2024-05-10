from flask import Flask, request, render_template, jsonify
import base64
import googlemaps
import time
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict
import requests
from dotenv import load_dotenv
import os


load_dotenv()

app = Flask(__name__)

api_key = os.getenv('OPENAI_API_KEY')
maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')

gmaps = googlemaps.Client(key=maps_api_key)
api_key = api_key
df = pd.read_excel('Locality add.xlsx')


multipliers = {
    'hindu_temple': 1,
    'church': 2.5,
    'mosque': 5,
    'synagogue': 1,
    'buddhist_temple': 1,
    'sikh_gurdwara': 1
}

religious_affiliation = {
    'hindu_temple': 'Hindu',
    'church': 'Christian',
    'mosque': 'Muslim',
    'synagogue': 'Jewish',
    'buddhist_temple': 'Buddhist',
    'sikh_gurdwara': 'Sikh'
}

def search_places_of_worship(address, radius=500):

    geocode_result = gmaps.geocode(address)
    if not geocode_result:
        return [], {}

    location = geocode_result[0]['geometry']['location']
    lat, lng = location['lat'], location['lng']
    query = "place of worship"


    results = gmaps.places_nearby(location=(lat, lng), radius=radius, keyword=query)
    places = []
    place_types = defaultdict(int)

    def process_results(results, places, place_types):
        for result in results['results']:
            places.append({
                'name': result.get('name'),
                'address': result.get('formatted_address'),
                'latitude': result['geometry']['location']['lat'],
                'longitude': result['geometry']['location']['lng']
            })
            for place_type in result.get('types', []):
                if place_type in ['church', 'mosque', 'synagogue', 'hindu_temple', 'buddhist_temple', 'sikh_gurdwara']:
                    place_types[place_type] += 1

    if results.get('results'):
        process_results(results, places, place_types)

        while 'next_page_token' in results:
            time.sleep(2)  
            results = gmaps.places_nearby(page_token=results['next_page_token'])
            process_results(results, places, place_types)

    return places, place_types



def extract_lat_lng(data):
    return [(item['geometry']['location']['lat'], item['geometry']['location']['lng']) for item in data]


def collect_places(places, results, seen):
    for result in results['results']:
        
        identifier = (result.get('name'), result.get('vicinity'))
        if identifier not in seen:
            seen.add(identifier)
            places.append({
                'name': result.get('name'),
                'address': result.get('vicinity'),
                'type': ', '.join(result.get('types', []))
            })

def search_places(query):
    types = ['grocery_or_supermarket']
    places = []
    seen = set()  

    for place_type in types:
        results = gmaps.places_nearby(location=query, radius=1000, type=place_type)
        collect_places(places, results, seen)
        while 'next_page_token' in results:
            time.sleep(2)  
            results = gmaps.places_nearby(page_token=results['next_page_token'])
            collect_places(places, results, seen)

    return len(places)

def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def find_nearest_mcdonalds(address):
            geocode_result = gmaps.geocode(address)

            location = geocode_result[0]['geometry']['location']
            lat, lng = location['lat'], location['lng']

            places_result = gmaps.places_nearby(location=(lat, lng), keyword='McDonald\'s', rank_by='distance')
            if places_result['results']:
                nearest_mcdonalds = places_result['results'][0]
                mcd_name = nearest_mcdonalds.get('name')
                mcd_namevicinity = nearest_mcdonalds.get('vicinity')
                res_name = f"The nearest McDonald's is {mcd_name} located at {mcd_namevicinity}."
            else:
                res_name= "No McDonald's found nearby."
            return res_name

def grid_search_places_of_worship(address, radius, step_size, grid_size):
            geocode_result = gmaps.geocode(address)
            if not geocode_result:
                print("Geocode result was not successful.")
                return

            location = geocode_result[0]['geometry']['location']
            base_lat, base_lng = location['lat'], location['lng']

            unique_places = set()
            for component in geocode_result[0]['address_components']:
                if 'sublocality' in component['types']:
                    unique_places.add(component['long_name'])
                    break

            for i in range(-grid_size, grid_size + 1):
                for j in range(-grid_size, grid_size + 1):
                    new_lat = base_lat + (i * step_size)
                    new_lng = base_lng + (j * step_size)
                    results = gmaps.places_nearby(location=(new_lat, new_lng), radius=radius, type=['neighborhood'])
                    for result in results['results']:
                        unique_places.add(result['name'])
            return unique_places

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        address = request.form['address']
        radius = 1000
        image = request.files['imageUpload']
        image_path = 'static/uploads/' + image.filename 
        image.save(image_path)
        encoded_image = encode_image(image_path)

        geocode_result = gmaps.geocode(address)
        if not geocode_result:
            return jsonify({'error': 'Address not found'})

        location = geocode_result[0]['geometry']['location']
        lat_lng = f"{location['lat']},{location['lng']}"
        
        places_count = search_places(lat_lng)

        locations = extract_lat_lng(geocode_result)

        for lat_lng in locations:
            df['Distance'] = df.apply(lambda row: haversine(lat_lng[1], lat_lng[0], row['latitude'], row['longitude']), axis=1)
        
        min_distance = df['Distance'].min()
        nearest_place = df[df['Distance'] <= min_distance]

        places, place_types = search_places_of_worship(address, radius)
        adjusted_totals = {ptype: count * multipliers.get(ptype, 1) for ptype, count in place_types.items()}
        total_adjusted = sum(adjusted_totals.values())
        estimated_population = int(nearest_place['pop_d'].values[0] * 2)
        demographics = {}
        for ptype, adjusted_count in adjusted_totals.items():
            percent = (adjusted_count / total_adjusted) * 100
            demographics[religious_affiliation[ptype]] = {
                'percent': percent,
                'estimated_population': (percent / 100) * estimated_population
            }

        majority_population = max(demographics.items(), key=lambda x: x[1]['estimated_population'])

        unique_places = list(grid_search_places_of_worship(address, radius=1000, step_size=0.005, grid_size=1))

        res_name = find_nearest_mcdonalds(address)

        base64_image = encode_image(image_path)

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"Is the place is right to start a retial shop here as the area information is given as the compitition is {places_count} display count and  Nearest locality is {nearest_place['locality'].values[0]}, Approx Population: {nearest_place['pop_d'].values[0] * 5}, No. of Families: {(nearest_place['pop_d'].values[0] * 5) // 4} Tier of the city: {nearest_place['tier'].values[0]} also The majority population is {majority_population[0]} in 1km. So also suggest something for productivity according the majority religion."        
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 3000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and response_data['choices']:
                chat_gpt_response = response_data['choices'][0]['message'] if 'message' in response_data['choices'][0] else "No message returned"
            else:
                chat_gpt_response = "No choices available in the response"
        else:
            chat_gpt_response = f"Failed to get a valid response from OpenAI, status code: {response.status_code}"

        # Prepare data for response
        response_data = {
            'address': address,
            'places_count': int(places_count),
            'nearest_locality': nearest_place['locality'].values[0],
            'population_estimate': int(nearest_place['pop_d'].values[0] * 5),
            'family_estimate': int(nearest_place['pop_d'].values[0] * 5 // 4),
            'tier_of_city': int(nearest_place['tier'].values[0]),
            'majority_religion': majority_population[0],
            'nearbyplces': unique_places,
            'Mcd': res_name,
            'ChatGpt_Response' : chat_gpt_response,
        }

        return jsonify(response_data)

    return render_template('index.html')

if __name__ == '__main__':
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    from werkzeug.wrappers import Response

    app_dispatch = DispatcherMiddleware(lambda e, s: Response(status=404), {
        '/api': app
    })

    def handler(event, context):
        return app_dispatch(event, context)
