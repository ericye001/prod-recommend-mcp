import os
import re
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import numpy as np

# Elasticsearch
from elasticsearch8 import AsyncElasticsearch
import asyncio

# Google Cloud
from google.cloud import storage
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

@dataclass
class ProductSpec:
    """Product specification data class"""
    product_id: str
    name: str
    url: str
    category: Optional[str]
    price: Optional[float]
    description: Optional[str]
    features: List[str]
    technical_specs: Dict[str, Any]
    image_urls: List[str]

class GofancoScraper:
    """Scraper for gofanco.com product specifications"""
    
    def __init__(self, base_url: str = "https://www.gofanco.com"):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Rotate user agents to avoid being blocked
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
        ]
        
        self.headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
        
        # Add a random delay between requests to avoid rate limiting
        self.min_delay = 1.0  # Minimum delay in seconds
        self.max_delay = 3.0  # Maximum delay in seconds
    
    def _get_random_user_agent(self):
        """Return a random user agent from the list"""
        import random
        return random.choice(self.user_agents)
    
    def _get_random_delay(self):
        """Return a random delay between min_delay and max_delay"""
        import random
        return random.uniform(self.min_delay, self.max_delay)
    
    def _make_request(self, url):
        """Make a request with error handling and rotating user agents"""
        # Update user agent before each request
        self.headers["User-Agent"] = self._get_random_user_agent()
        
        try:
            # Make the request
            response = self.session.get(url, headers=self.headers, timeout=10)
            
            # If we got a 403 or 429, possibly being rate limited
            if response.status_code in (403, 429):
                print(f"Received status code {response.status_code}, possibly rate limited. Waiting 30 seconds...")
                time.sleep(30)  # Longer delay for rate limiting
                
                # Try again with a different user agent
                self.headers["User-Agent"] = self._get_random_user_agent()
                response = self.session.get(url, headers=self.headers, timeout=10)
            
            # Simulate human reading time - wait between requests
            time.sleep(self._get_random_delay())
            
            return response
        except Exception as e:
            print(f"Error making request to {url}: {str(e)}")
            return None
    
    def get_product_urls(self, max_pages: int = 10) -> List[str]:
        """Get all product URLs from product listing pages"""
        product_urls = []
        
        # Categories to scrape - updated with correct gofanco.com URL structure
        categories = [
            # Main product categories
            "/products/hdmi-extenders.html",
            "/products/splitters.html", 
            "/products/switches-matrixes.html",
            "/products/installer-gear.html",
            "/products/cables-adapters.html",
            "/products/scalers-converters.html",
            "/products/video-wall.html",
            "/products/audio.html",
            "/products/kvm.html",
            "/products/capture-devices.html",
            "/products/charging-stations.html",
            "/products/fiber-optic.html",
            
            # HDMI Extender subcategories
            "/products/hdmi-extenders/over-ethernet.html",
            "/products/hdmi-extenders/wireless.html",
            "/products/hdmi-extenders/over-ip.html",
            "/products/hdmi-extenders/hdbaset.html",
            "/products/hdmi-extenders/over-coaxial.html",
            "/products/hdmi-extenders/over-fiber.html",
            "/products/hdmi-extenders/extender-splitters.html",
            "/products/hdmi-extenders/repeater.html",
            
            # Fallback to all products
            "/products/all.html"
        ]
        
        for category in categories:
            page = 1
            while page <= max_pages:
                # All categories now end with .html, so we can use the same pagination logic
                url = f"{self.base_url}{category}"
                if page > 1:
                    # For pages beyond first page, add /?p=N to the URL
                    url = url + f"?p={page}"
                    
                print(f"Fetching category page: {url}")
                
                try:
                    response = self._make_request(url)
                    
                    if not response or response.status_code != 200:
                        status_code = response.status_code if response else "No response"
                        print(f"Failed to fetch category page: {url}, status code: {status_code}")
                        break
                        
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Try multiple possible selectors for product links
                    product_links = []
                    selectors_to_try = [
                        # Gofanco specific selectors based on actual site structure
                        "li.item.product.product-item a.product-item-link",
                        ".products-grid .product-item-link", 
                        ".product-item-info a.product-item-link",
                        ".product.details.product-item-details a.product-item-link",
                        # Generic fallback selectors
                        ".product-item a",
                        ".products-grid a.product-link",
                        ".item.product a[href*='/products/']",
                        "a[href*='/products/']",
                        # Extremely generic fallback
                        "a[href]:not([href^='#']):not([href^='javascript'])"
                    ]
                    
                    for selector in selectors_to_try:
                        links = soup.select(selector)
                        if links:
                            # Print information about the links found
                            print(f"Found {len(links)} product links with selector: {selector}")
                            
                            # Show sample links for debugging
                            if len(links) > 0:
                                sample_link = links[0].get("href", "")
                                print(f"Sample link: {sample_link}")
                                
                            product_links.extend(links)
                            break
                    
                    if not product_links:
                        print(f"No product links found on page {page} of {category}. Trying to save HTML for debugging...")
                        # Save the HTML for inspection (helpful for debugging)
                        debug_dir = "debug"
                        os.makedirs(debug_dir, exist_ok=True)
                        with open(f"{debug_dir}/category_page_{category.replace('/', '_')}_{page}.html", "w", encoding="utf-8") as f:
                            f.write(response.text)
                        print(f"HTML saved for debugging")
                        break
                        
                    for link in product_links:
                        product_url = link.get("href")
                        if product_url:
                            if not product_url.startswith("http"):
                                product_url = self.base_url + product_url
                            product_urls.append(product_url)
                    
                    page += 1
                except Exception as e:
                    print(f"Error fetching category page {url}: {str(e)}")
                    break
        
        unique_urls = list(set(product_urls))  # Remove duplicates
        print(f"Found {len(unique_urls)} unique product URLs")
        return unique_urls
    
    def extract_product_specs(self, url: str) -> Optional[ProductSpec]:
        """Extract product specifications from a product page"""
        try:
            print(f"Extracting product specs from: {url}")
            response = self._make_request(url)
            
            if not response or response.status_code != 200:
                status_code = response.status_code if response else "No response"
                print(f"Failed to fetch product page: {url}, status code: {status_code}")
                return None
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract basic product info - adjust selectors based on actual website structure
            product_id = self._extract_product_id(soup, url)
            name = self._extract_product_name(soup)
            category = self._extract_category(soup)
            price = self._extract_price(soup)
            description = self._extract_description(soup)
            features = self._extract_features(soup)
            technical_specs = self._extract_technical_specs(soup)
            image_urls = self._extract_image_urls(soup)
            
            print(f"Extracted product: {name} (ID: {product_id})")
            if technical_specs:
                print(f"  Found {len(technical_specs)} technical specifications")
            if features:
                print(f"  Found {len(features)} features")
            
            return ProductSpec(
                product_id=product_id,
                name=name,
                url=url,
                category=category,
                price=price,
                description=description,
                features=features,
                technical_specs=technical_specs,
                image_urls=image_urls
            )
            
        except Exception as e:
            print(f"Error extracting data from {url}: {str(e)}")
            return None
    
    def _extract_product_id(self, soup: BeautifulSoup, url: str) -> str:
        """Extract product ID from soup or URL"""
        # Try to find product ID in meta tags or other elements
        try:
            # Look for SKU or product ID in page
            sku_elem = soup.select_one("[itemprop='sku']")
            if sku_elem:
                return sku_elem.text.strip()
            
            # Extract from URL if possible
            match = re.search(r'products/([^/?]+)', url)
            if match:
                return match.group(1)
        except:
            pass
        
        # Fallback to URL hash
        return str(hash(url))
    
    def _extract_product_name(self, soup: BeautifulSoup) -> str:
        """Extract product name"""
        try:
            # Try multiple selectors for product name (gofanco specific)
            selectors = [
                "h1.page-title",
                ".product-info-main h1.page-title",
                ".page-title-wrapper h1",
                ".product-name h1",
                # Generic fallbacks
                "h1.product-title",
                "h1[itemprop='name']",
                "h1"
            ]
            
            for selector in selectors:
                name_elem = soup.select_one(selector)
                if name_elem:
                    return name_elem.text.strip()
        except Exception as e:
            print(f"Error extracting product name: {str(e)}")
        return "Unknown Product"
    
    def _extract_category(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract product category"""
        try:
            # Try multiple selectors for breadcrumbs (gofanco specific)
            breadcrumb_selectors = [
                ".breadcrumbs li.item",
                ".breadcrumb-container li",
                ".breadcrumbs ul li",
                # Generic fallbacks
                "nav.breadcrumb a",
                ".breadcrumbs a"
            ]
            
            for selector in breadcrumb_selectors:
                breadcrumbs = soup.select(selector)
                if breadcrumbs and len(breadcrumbs) > 1:
                    # Second-to-last breadcrumb is usually the category
                    return breadcrumbs[-2].text.strip()
                    
            # If no breadcrumb found, try to extract from URL
            canonical_link = soup.select_one("link[rel='canonical']")
            if canonical_link:
                href = canonical_link.get('href', '')
                # Extract category from URL pattern like /products/category/product.html
                match = re.search(r'/products/([^/]+)/', href)
                if match:
                    return match.group(1).replace('-', ' ').title()
        except Exception as e:
            print(f"Error extracting category: {str(e)}")
        return None
    
    def _extract_price(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract product price"""
        try:
            # Try multiple selectors for price (gofanco specific)
            price_selectors = [
                ".price-container .price",
                ".product-info-price .price-container .price",
                ".product-info-price .price",
                "[data-price-type='finalPrice'] .price",
                ".special-price .price",
                # Generic fallbacks
                "[itemprop='price']",
                ".price",
                ".price-item--regular"
            ]
            
            for selector in price_selectors:
                price_elem = soup.select_one(selector)
                if price_elem:
                    price_text = price_elem.text.strip()
                    # Remove currency symbol and commas, then convert to float
                    price_text = re.sub(r'[^\d.]', '', price_text)
                    if price_text:
                        return float(price_text)
        except Exception as e:
            print(f"Error extracting price: {str(e)}")
        return None
    
    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract product description"""
        try:
            # Try multiple selectors for description (gofanco specific)
            desc_selectors = [
                ".product.attribute.description .value",
                ".product.attribute.overview .value",
                ".product-info-main .description",
                ".product-info-main .product.attribute.overview",
                ".product-info-main [itemprop='description']",
                # Generic fallbacks
                "[itemprop='description']",
                ".product-description"
            ]
            
            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem:
                    return desc_elem.text.strip()
        except Exception as e:
            print(f"Error extracting description: {str(e)}")
        return None
    
    def _extract_features(self, soup: BeautifulSoup) -> List[str]:
        """Extract product features"""
        features = []
        try:
            # Look for features in product attributes (gofanco specific)
            feature_selectors = [
                ".product.attribute.features ul li",
                ".product.attribute.feature ul li",
                ".additional-attributes tr",
                ".product-features ul li",
                ".product-features li",
                # Check for "Features" sections or tabs
                "[data-tab='features'] li",
                ".features-list li"
            ]
            
            for selector in feature_selectors:
                feature_items = soup.select(selector)
                if feature_items:
                    for item in feature_items:
                        if selector == ".additional-attributes tr":
                            # Handle table row format
                            cells = item.select("th, td")
                            if len(cells) >= 1:
                                feature_text = " ".join([cell.text.strip() for cell in cells])
                                if feature_text:
                                    features.append(feature_text)
                        else:
                            # Handle list item format
                            feature_text = item.text.strip()
                            if feature_text:
                                features.append(feature_text)
                    if features:
                        break  # Once we find features with one selector, stop
            
            # If no features found through conventional selectors, try to find any feature section
            if not features:
                # Look for any section that might have features
                feature_sections = []
                for heading in soup.find_all(["h2", "h3", "h4", "h5", "strong"]):
                    heading_text = heading.text.lower().strip()
                    if "feature" in heading_text or "key" in heading_text:
                        # Found a potential features section heading
                        feature_sections.append(heading)
                
                # Process each feature section
                for heading in feature_sections:
                    # Look for lists in the next sibling elements
                    for sibling in heading.find_next_siblings():
                        if sibling.name in ["ul", "ol"]:
                            for li in sibling.find_all("li"):
                                feature_text = li.text.strip()
                                if feature_text:
                                    features.append(feature_text)
                            if features:
                                break
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
        return features
    
    def _extract_technical_specs(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract technical specifications"""
        specs = {}
        try:
            # Try to find spec tables (gofanco specific)
            spec_table_selectors = [
                ".additional-attributes-wrapper table",
                ".product.attribute.specs table",
                ".product.attribute.specifications table",
                ".data.table.additional-attributes",
                "#product-attribute-specs-table",
                # Generic fallbacks
                "table.specifications",
                ".specs-table",
                ".technical-specs table"
            ]
            
            for selector in spec_table_selectors:
                specs_table = soup.select_one(selector)
                if specs_table:
                    rows = specs_table.select("tr")
                    for row in rows:
                        # Handle different table cell types (th/td)
                        th = row.select_one("th")
                        td = row.select_one("td")
                        if th and td:
                            key = th.text.strip()
                            value = td.text.strip()
                            if key and value:  # Skip empty rows
                                specs[key] = value
                    if specs:  # If we found specs with this selector, break
                        break
            
            # Look for spec sections with definition lists
            if not specs:
                dl_selectors = [".product-specs dl", ".specifications dl", ".specs dl"]
                for selector in dl_selectors:
                    dl = soup.select_one(selector)
                    if dl:
                        dt_elements = dl.select("dt")
                        dd_elements = dl.select("dd")
                        if len(dt_elements) == len(dd_elements):
                            for i in range(len(dt_elements)):
                                key = dt_elements[i].text.strip()
                                value = dd_elements[i].text.strip()
                                if key and value:
                                    specs[key] = value
                        if specs:
                            break
            
            # Look for specs in product attributes with labeled sections
            if not specs:
                # Find all the product attribute blocks
                attribute_blocks = soup.select(".product.attribute")
                for block in attribute_blocks:
                    # Look for a label/caption/heading in this block
                    label_elem = block.select_one(".attribute-label, .heading, caption, h2, h3, h4")
                    if label_elem:
                        label = label_elem.text.strip().rstrip(':')
                        # Skip certain labels which are descriptions, not specs
                        if label.lower() in ['description', 'overview', 'features']:
                            continue
                        # Get the content
                        value_elem = block.select_one(".attribute-value, .value, p")
                        if value_elem:
                            value = value_elem.text.strip()
                            if label and value:
                                specs[label] = value
        except Exception as e:
            print(f"Error extracting technical specs: {str(e)}")
        return specs
    
    def _extract_image_urls(self, soup: BeautifulSoup) -> List[str]:
        """Extract product image URLs"""
        image_urls = []
        try:
            # Try multiple selectors for product images (gofanco specific)
            image_selectors = [
                ".gallery-placeholder img",
                ".product.media img",
                ".fotorama__img",
                ".fotorama__stage img",
                "[role='option'] img",
                ".gallery-images img",
                # Generic fallbacks
                ".product-image img",
                ".product-gallery img",
                "[data-product-media-type='image'] img"
            ]
            
            for selector in image_selectors:
                image_elements = soup.select(selector)
                if image_elements:
                    for img in image_elements:
                        # Try different image attributes
                        for attr in ["src", "data-lazy", "data-src", "data-full", "data-srcset", "data-original"]:
                            src = img.get(attr)
                            if src:
                                # Clean up srcset formats
                                if "," in src:
                                    src = src.split(",")[0].split(" ")[0]
                                # Make sure URL is absolute
                                if not src.startswith("http"):
                                    src = self.base_url + src
                                # Avoid duplicates
                                if src not in image_urls:
                                    image_urls.append(src)
                                break
                    if image_urls:
                        break  # Once we find images with one selector, stop
        except Exception as e:
            print(f"Error extracting image URLs: {str(e)}")
        return image_urls

class ElasticsearchIndexer:
    """Class to handle Elasticsearch indexing"""
    
    def __init__(self, es_url: str, bearer_token: str, cloud_id: str, index_name: str = "gofanco_products"):
        self.es_url = es_url
        self.bearer_token = bearer_token
        self.cloud_id = cloud_id
        self.index_name = index_name
        self.es_client = None
    
    async def connect(self):
        """Connect to Elasticsearch"""
        self.es_client = AsyncElasticsearch(
            hosts=self.es_url,
            api_key=self.bearer_token,
            verify_certs=True,
        )
    
    async def close(self):
        """Close Elasticsearch connection"""
        if self.es_client:
            await self.es_client.close()
    
    async def create_index(self):
        """Create index with appropriate mappings"""
        # Define mapping for products
        mapping = {
            "mappings": {
                "properties": {
                    "product_id": {"type": "keyword"},
                    "name": {"type": "text"},
                    "url": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "price": {"type": "float"},
                    "description": {"type": "text"},
                    "features": {"type": "text"},
                    "technical_specs": {"type": "object", "enabled": True},
                    "image_urls": {"type": "keyword"},
                    # Add additional fields as needed
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            }
        }
        
        # Check if index exists
        if not await self.es_client.indices.exists(index=self.index_name):
            # Create index with mapping
            await self.es_client.indices.create(index=self.index_name, body=mapping)
            print(f"Created Elasticsearch index: {self.index_name}")
    
    async def index_product(self, product: ProductSpec):
        """Index a single product"""
        product_dict = asdict(product)
        
        # Convert technical specs to a format that's easier to query
        flattened_specs = {}
        for key, value in product_dict["technical_specs"].items():
            # Create a valid field name by replacing invalid characters
            field_name = re.sub(r'[^\w]', '_', key).lower()
            flattened_specs[field_name] = value
        
        product_dict["technical_specs"] = flattened_specs
        
        # Index document
        await self.es_client.index(
            index=self.index_name,
            id=product_dict["product_id"],
            document=product_dict
        )
    
    async def index_products(self, products: List[ProductSpec]):
        """Index multiple products using bulk API"""
        if not products:
            return
            
        # Connect if not already connected
        if not self.es_client:
            await self.connect()
            
        # Create index if it doesn't exist
        await self.create_index()
        
        # Prepare bulk indexing operations
        operations = []
        for product in products:
            # Convert to dictionary
            product_dict = asdict(product)
            
            # Flatten technical specs
            flattened_specs = {}
            for key, value in product_dict["technical_specs"].items():
                field_name = re.sub(r'[^\w]', '_', key).lower()
                flattened_specs[field_name] = value
            
            product_dict["technical_specs"] = flattened_specs
            
            # Add to bulk operations
            operations.append({"index": {"_index": self.index_name, "_id": product_dict["product_id"]}})
            operations.append(product_dict)
        
        # Execute bulk operation
        if operations:
            await self.es_client.bulk(operations=operations)
            print(f"Indexed {len(products)} products in Elasticsearch")

class GoogleCloudVectorStore:
    """Class to handle Google Cloud Vector Store operations"""
    
    def __init__(self, project_id: str, location: str, embedding_model_name: str = "textembedding-gecko@003"):
        self.project_id = project_id
        self.location = location
        self.embedding_model_name = embedding_model_name
        
        # Initialize Google Cloud clients
        aiplatform.init(project=project_id, location=location)
        self.storage_client = storage.Client(project=project_id)
        
        # Initialize Vertex AI embedding model
        self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using Vertex AI"""
        embeddings = []
        
        # Process in batches to avoid API limits
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Get embeddings for batch
            batch_embeddings = self.embedding_model.get_embeddings(batch)
            
            # Extract values
            for embedding in batch_embeddings:
                embeddings.append(embedding.values)
            
            # Be nice to the API
            if i + batch_size < len(texts):
                time.sleep(1)
        
        return embeddings
    
    def create_product_embeddings(self, products: List[ProductSpec]) -> List[Dict[str, Any]]:
        """Create embeddings for products"""
        # Create documents with text to embed
        documents = []
        texts_to_embed = []
        
        for product in products:
            # Combine product info into a single text for embedding
            text_to_embed = f"{product.name} {product.description or ''} "
            
            # Add features
            if product.features:
                text_to_embed += " ".join(product.features) + " "
            
            # Add technical specs
            if product.technical_specs:
                spec_text = " ".join([f"{k}: {v}" for k, v in product.technical_specs.items()])
                text_to_embed += spec_text
            
            # Keep track of original product and text
            documents.append({
                "product_id": product.product_id,
                "text": text_to_embed,
                "product": asdict(product)
            })
            
            texts_to_embed.append(text_to_embed)
        
        # Get embeddings
        print(f"Generating embeddings for {len(texts_to_embed)} products...")
        embeddings = self.get_embeddings(texts_to_embed)
        
        # Combine embeddings with documents
        for i, embedding in enumerate(embeddings):
            documents[i]["embedding"] = embedding
        
        return documents
    
    def save_to_bigquery(self, product_embeddings: List[Dict[str, Any]], dataset_id: str, table_id: str):
        """Save product embeddings to BigQuery (for Vertex AI Vector Search)"""
        from google.cloud import bigquery
        
        # Initialize BigQuery client
        bq_client = bigquery.Client(project=self.project_id)
        
        # Create dataset if it doesn't exist
        dataset_ref = bq_client.dataset(dataset_id)
        try:
            bq_client.get_dataset(dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = self.location
            bq_client.create_dataset(dataset, exists_ok=True)
        
        # Define schema
        schema = [
            bigquery.SchemaField("product_id", "STRING"),
            bigquery.SchemaField("name", "STRING"),
            bigquery.SchemaField("url", "STRING"),
            bigquery.SchemaField("category", "STRING"),
            bigquery.SchemaField("price", "FLOAT"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("features", "STRING", mode="REPEATED"),
            bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
            # Technical specs will be stored as JSON
            bigquery.SchemaField("technical_specs", "STRING")
        ]
        
        # Create table
        table_ref = dataset_ref.table(table_id)
        table = bigquery.Table(table_ref, schema=schema)
        
        try:
            bq_client.get_table(table)
        except Exception:
            table = bq_client.create_table(table, exists_ok=True)
        
        # Prepare rows for insertion
        rows = []
        for doc in product_embeddings:
            product = doc["product"]
            row = {
                "product_id": product["product_id"],
                "name": product["name"],
                "url": product["url"],
                "category": product["category"],
                "price": product["price"],
                "description": product["description"],
                "features": product["features"],
                "embedding": doc["embedding"],
                "technical_specs": json.dumps(product["technical_specs"])
            }
            rows.append(row)
        
        # Insert rows
        errors = bq_client.insert_rows_json(table, rows)
        if errors:
            print(f"Errors inserting rows: {errors}")
        else:
            print(f"Inserted {len(rows)} rows into BigQuery")
    
    def create_vector_search_index(self, dataset_id: str, table_id: str, index_id: str):
        """Create Vector Search index for the data in BigQuery"""
        # Note: This requires Vertex AI VectorSearch service to be enabled
        
        # Create VectorSearch Index on the BigQuery table
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=f"{index_id}-endpoint",
            project=self.project_id,
            location=self.location
        )
        
        # Define index configuration for text embeddings
        dimensions = 768  # Depends on the embedding model used
        
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=index_id,
            project=self.project_id,
            location=self.location,
            dimensions=dimensions,
            approximate_neighbors_count=10,
            distance_measure_type="DOT_PRODUCT_DISTANCE",
            leaf_node_embedding_count=500,
            leaf_nodes_to_search_percent=10,
            description="Vector search index for gofanco product specs"
        )
        
        # Create BigQuery index source
        bq_source = f"bq://{self.project_id}.{dataset_id}.{table_id}"
        
        # Create index
        print(f"Creating vector search index {index_id} from {bq_source}...")
        index.create_from_bigquery(
            bigquery_uri=bq_source,
            embedding_column_name="embedding",
            id_column_name="product_id"
        )
        
        # Deploy index to endpoint
        print(f"Deploying index {index_id} to endpoint...")
        index_endpoint.deploy_index(
            index=index,
            deployed_index_id=index_id
        )
        
        print(f"Vector search index deployed successfully: {index_id}")
        return index_endpoint, index

async def main():
    try:
        # Configuration
        ES_URL = os.getenv("ES_URL", "https://af49f69415bb43a0be33665110ab74e0.us-central1.gcp.cloud.es.io:443")
        BEARER_TOKEN = os.getenv("ES_BEARER_TOKEN", "ZEN4SENaWUI3VWdiYjFtdEdMUEY6Q2l5VFJUdEJUSnlpUy1aeEo0QjhEZw==")
        CLOUD_ID = os.getenv("ES_CLOUD_ID", "My_deployment:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGFmNDlmNjk0MTViYjQzYTBiZTMzNjY1MTEwYWI3NGUwJDI5OTVlODFhZjk3MTRiMGNhODliYTJhNDVlNGE1ZWZi")
        
        GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "customer-service-agent")
        GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        # Check for required environment variables
        if ES_URL == "https://your-es-proxy-url" or BEARER_TOKEN == "your-bearer-token":
            print("WARNING: Using default Elasticsearch configuration. Set ES_URL and ES_BEARER_TOKEN environment variables.")
        
        if GOOGLE_CLOUD_PROJECT == "your-gcp-project":
            print("WARNING: Using default Google Cloud project. Set GOOGLE_CLOUD_PROJECT environment variable.")
        
        # Initialize scraper
        scraper = GofancoScraper()
        
        scrape_flag = False
        
        if scrape_flag:
            print("Getting product URLs...")
            product_urls = scraper.get_product_urls(max_pages=50)
            print(f"Found {len(product_urls)} product URLs")
            
            if not product_urls:
                print("No product URLs found. Check the selectors and website structure.")
                return
            
            # Extract product specs
            print("Extracting product specifications...")
            products = []
            for url in tqdm(product_urls[:-1]):  # Start with just 10 products for testing
                product = scraper.extract_product_specs(url)
                if product:
                    products.append(product)
            
            print(f"Extracted specifications for {len(products)} products")
            
            if not products:
                print("No products extracted. Check the selectors and website structure.")
                return
            
            # Save products to a JSON file (for backup)
            os.makedirs("data", exist_ok=True)
            with open("data/products.json", "w") as f:
                json.dump([asdict(p) for p in products], f, indent=2)
            print(f"Saved {len(products)} products to data/products.json")
        
        else:
            with open("data/products.json", "r") as f:
                products_data = json.load(f)
                products = [ProductSpec(**product) for product in products_data]
        
        # Initialize Elasticsearch indexer
        es_indexer = ElasticsearchIndexer(es_url=ES_URL, bearer_token=BEARER_TOKEN, cloud_id=CLOUD_ID, index_name="gofanco_products")
        
        # Index products in Elasticsearch
        print("Indexing products in Elasticsearch...")
        try:
            await es_indexer.connect()
            await es_indexer.index_products(products)
        except Exception as e:
            print(f"Error indexing products in Elasticsearch: {str(e)}")
        finally:
            await es_indexer.close()
        
        # Initialize Google Cloud Vector Store
        try:
            gcp_vector_store = GoogleCloudVectorStore(
                project_id=GOOGLE_CLOUD_PROJECT,
                location=GOOGLE_CLOUD_LOCATION
            )
            
            # Create product embeddings
            print("Creating product embeddings...")
            product_embeddings = gcp_vector_store.create_product_embeddings(products)
            
            # Save to BigQuery (required for Vertex AI Vector Search)
            print("Saving data to BigQuery...")
            gcp_vector_store.save_to_bigquery(
                product_embeddings=product_embeddings,
                dataset_id="gofanco_products",
                table_id="product_embeddings"
            )
            
            # Create vector search index
            print("Creating vector search index...")
            gcp_vector_store.create_vector_search_index(
                dataset_id="gofanco_products",
                table_id="product_embeddings",
                index_id="gofanco-products-index"
            )
        except Exception as e:
            print(f"Error in Google Cloud operations: {str(e)}")
        
        print("Process completed successfully!")
    except Exception as e:
        print(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())