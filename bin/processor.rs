// embedding_processor/src/main.rs

use env_logger;
use futures::stream::{self, StreamExt};
use log::{error, info};
use reqwest::Client;
use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};

// Struct representing a post from the source database
#[derive(Debug, Serialize, Clone)]
struct Post {
    id: i32,
    rss_url: String,
    title: String,
    link: String,
    published: Option<String>,
    description: Option<String>,
}

// Struct representing an embedded post for the target database
#[derive(Debug, Serialize)]
struct EmbeddedPost {
    id: i32,
    rss_url: String,
    title: String,
    link: String,
    published: Option<String>,
    description: Option<String>,
    embedding: Vec<f32>,
}

#[tokio::main]
async fn main() {
    // Initialize the logger
    env_logger::init();

    // Define paths to the databases
    let source_db_path = "assets/rss_posts.db";
    let target_db_path = "assets/embedded_posts.db";

    // Initialize the target SQLite database
    if let Err(e) = initialize_target_database(target_db_path) {
        error!("Failed to initialize target database: {}", e);
        return;
    }

    // Initialize the HTTP client
    let client = Client::builder()
        .user_agent("EmbeddingProcessor/1.0")
        .build()
        .expect("Failed to build HTTP client");

    let client = Arc::new(client);

    // Define rate limiting: e.g., 5 requests per second
    let rate_limit = 5;
    let semaphore = Arc::new(Semaphore::new(rate_limit));

    // Fetch all posts from the source database
    let posts = match fetch_all_posts(source_db_path) {
        Ok(posts) => posts,
        Err(e) => {
            error!("Failed to fetch posts from source database: {}", e);
            return;
        }
    };

    if posts.is_empty() {
        info!("No posts found in source database.");
        return;
    }

    info!("Found {} posts to process.", posts.len());

    // Process posts concurrently with rate limiting
    let tasks = stream::iter(posts)
        .map(|post| {
            let client = Arc::clone(&client);
            let semaphore = Arc::clone(&semaphore);
            let target_db_path = target_db_path.to_string();
            async move {
                let _permit = semaphore.acquire().await.unwrap();
                match process_post(&client, &target_db_path, post.clone()).await {
                    Ok(_) => info!("Processed post: {}", post.title),
                    Err(e) => error!("Error processing post {}: {}", post.title, e),
                }
                // Sleep to maintain the rate limit
                sleep(Duration::from_secs_f32(1.0 / rate_limit as f32)).await;
            }
        })
        .buffer_unordered(rate_limit);

    tasks.collect::<Vec<()>>().await;

    info!("Embedding processing complete.");
}

// Function to initialize the target SQLite database
fn initialize_target_database(db_path: &str) -> SqlResult<()> {
    let conn = Connection::open(db_path)?;

    // Enable Write-Ahead Logging for better concurrency
    conn.execute_batch("PRAGMA journal_mode = WAL;")?;

    // Create the embedded_posts table if it doesn't exist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embedded_posts (
            id INTEGER PRIMARY KEY,
            rss_url TEXT NOT NULL,
            title TEXT NOT NULL,
            link TEXT NOT NULL UNIQUE,
            published TEXT,
            description TEXT,
            embedding TEXT NOT NULL
        )",
        [],
    )?;
    Ok(())
}

// Function to fetch all posts from the source database
fn fetch_all_posts(db_path: &str) -> SqlResult<Vec<Post>> {
    let conn = Connection::open(db_path)?;
    let mut stmt =
        conn.prepare("SELECT id, rss_url, title, link, published, description FROM posts")?;
    let post_iter = stmt.query_map([], |row| {
        Ok(Post {
            id: row.get(0)?,
            rss_url: row.get(1)?,
            title: row.get(2)?,
            link: row.get(3)?,
            published: row.get(4)?,
            description: row.get(5)?,
        })
    })?;

    let mut posts = Vec::new();
    for post in post_iter {
        posts.push(post?);
    }
    Ok(posts)
}

// Function to process a single post: fetch embedding and store in target DB
async fn process_post(
    client: &Client,
    target_db_path: &str,
    post: Post,
) -> Result<(), Box<dyn std::error::Error>> {
    // Extract the description as the content to embed
    let content = match &post.description {
        Some(desc) => desc.clone(),
        None => {
            return Err("Post does not have a description".into());
        }
    };

    // Fetch embedding from Ollama's API
    let embedding = match get_embedding(client, &content).await {
        Ok(embed) => embed,
        Err(e) => {
            return Err(format!("Failed to get embedding: {}", e).into());
        }
    };

    // Serialize the embedding as JSON
    let embedding_json = serde_json::to_string(&embedding)?;

    // Insert into the target database
    let result = insert_embedded_post(target_db_path, &post, &embedding_json).await;

    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Failed to insert into target DB: {}", e).into()),
    }
}

// Function to insert an embedded post into the target database
async fn insert_embedded_post(db_path: &str, post: &Post, embedding_json: &str) -> SqlResult<()> {
    // Open a new connection for each insertion to ensure thread safety
    let conn = Connection::open(db_path)?;

    // Insert the embedded post
    conn.execute(
        "INSERT OR IGNORE INTO embedded_posts (id, rss_url, title, link, published, description, embedding) 
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            post.id,
            post.rss_url,
            post.title,
            post.link,
            post.published,
            post.description,
            embedding_json,
        ],
    )?;
    Ok(())
}

// Function to get embedding from Ollama's API
async fn get_embedding(
    client: &Client,
    text: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Define the Ollama API endpoint for embeddings
    let api_url = "http://localhost:11434/v1/embeddings"; // Adjust based on Ollama's API

    // Create the request payload
    let payload = json!({
        "model": "mxbai-embed-large", // Adjust based on the available models in Ollama
        "input": text,
    });

    // Send the POST request
    let response = client.post(api_url).json(&payload).send().await?;

    if !response.status().is_success() {
        return Err(format!("Ollama API returned status: {}", response.status()).into());
    }

    // Parse the response JSON
    let response_json: serde_json::Value = response.json().await?;
    let response_json = response_json.get("data").ok_or("Invalid response format")?;
    let response_json = response_json.get(0).ok_or("Invalid response format")?;

    // Extract the embedding vector
    // Adjust based on Ollama's API response structure
    if let Some(embedding) = response_json.get("embedding").and_then(|e| e.as_array()) {
        let vec: Vec<f32> = embedding
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        Ok(vec)
    } else {
        Err("Invalid embedding format in response".into())
    }
}
