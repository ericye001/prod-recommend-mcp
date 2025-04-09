import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from elasticsearch8 import AsyncElasticsearch
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gofanco Product Recommendation API")

# Elasticsearch configuration
ES_URL = os.getenv("ES_URL")
BEARER_TOKEN = os.getenv("ES_BEARER_TOKEN")
CLOUD_ID = os.getenv("ES_CLOUD_ID")

# Initialize Elasticsearch client
es_client = AsyncElasticsearch(
    hosts=ES_URL,
    api_key=BEARER_TOKEN,
    verify_certs=True,
)

class ProductQuery(BaseModel):
    query: str
    #category: str = None
    max_results: int = 5

class ProductResponse(BaseModel):
    product_id: str
    name: str
    url: str
    category: str
    description: str
    features: List[str]
    technical_specs: Dict[str, Any]
    score: float

@app.get("/health")
async def health_check():
    """Check Elasticsearch connection and index status"""
    try:
        # Check if we can connect to Elasticsearch
        info = await es_client.info()
        logger.info(f"Connected to Elasticsearch cluster: {info['cluster_name']}")
        
        # Check if our index exists
        index_exists = await es_client.indices.exists(index="gofanco_products")
        if not index_exists:
            return {"status": "error", "message": "Index 'gofanco_products' does not exist"}
        
        # Get index stats
        stats = await es_client.indices.stats(index="gofanco_products")
        doc_count = stats["_all"]["primaries"]["docs"]["count"]
        
        return {
            "status": "healthy",
            "cluster_name": info["cluster_name"],
            "index_exists": True,
            "document_count": doc_count
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/recommend", response_model=List[ProductResponse])
async def recommend_products(query: ProductQuery):
    """
    Recommend Gofanco products based on user query.
    """
    try:
        logger.info(f"Received recommendation request: {query}")
        
        # Build the search query
        search_query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query.query,
                            "fields": ["name^3", "description^2", "features", "technical_specs"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    }
                ]
            }
        }

        # Add category filter if specified
        #if query.category:
        #    search_query["bool"]["filter"] = [
        #        {"term": {"category.keyword": query.category}}
        #    ]

        logger.info(f"Executing search query: {search_query}")
        
        # Execute search with new syntax
        response = await es_client.search(
            index="gofanco_products",
            query=search_query,
            size=query.max_results
        )

        logger.info(f"Search returned {len(response['hits']['hits'])} results")
        
        # Process and return results
        recommendations = []
        for hit in response["hits"]["hits"]:
            product = hit["_source"]
            recommendations.append(ProductResponse(
                product_id=product["product_id"],
                name=product["name"],
                url=product["url"],
                category="ALL",
                description=product["description"],
                features=product["features"],
                technical_specs=product["technical_specs"],
                score=hit["_score"]
            ))

        return recommendations

    except Exception as e:
        logger.error(f"Error in recommend_products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    """
    Get list of available product categories.
    """
    try:
        logger.info("Fetching categories - Current index: gofanco_products but no category value in the index right now!!!")
        
        response = await es_client.search(
            index="gofanco_products",
            query={"match_all": {}},
            size=0,
            aggs={
                "categories": {
                    "terms": {
                        "field": "category.keyword",
                        "size": 100
                    }
                }
            }
        )
        
        logger.info(f"Categories response: {response}")
        
        if "aggregations" not in response:
            logger.error("No aggregations in response")
            return {"categories": []}
            
        categories = [bucket["key"] for bucket in response["aggregations"]["categories"]["buckets"]]
        logger.info(f"Found {len(categories)} categories")
        return {"categories": categories}
    
    except Exception as e:
        logger.error(f"Error in get_categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 