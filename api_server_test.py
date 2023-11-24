"""

API server test

"""

import pytest
from fastapi.testclient import TestClient
from api_server import app


@pytest.fixture(scope='module')
def client():
  
    """
    get dataset

    """

    api_client = TestClient(app)

    return api_client
  
  

def test_get(client):
  
    response = client.get('/')

    assert response.status_code == 200
    assert response.json() == {"message":"Greetings!"}


def test_get_malformed(client):
    response = client.get('/wrong_url')
    assert response.status_code != 200


def test_post_lower(client):
    response = client.post("/",
                            json= {
         "age": 19,
        "workclass": "Private",
        "education": "HS-grad",
        "maritalStatus": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
        
                            })
    
    assert response.status_code == 200
    assert response.json() == {'prediction':'<=50K'}


def test_post_higher(client):
    response = client.post("/",
                            json= {
        "age": 32,
        "workclass": "Private",
        "education": "Some-college",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States"
                            })
    
    assert response.status_code == 200
    assert response.json() == {'prediction':'>50K'}












