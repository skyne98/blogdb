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

// Struct representing a chunk of text for embedding
#[derive(Debug, Serialize)]
struct Chunk {
    article_id: i32,
    chunk_id: i32,
    text: String,
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

    // Define rate limiting: e.g., 100 requests per second
    let rate_limit = 100;
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

    // Ensure unique ids for each post
    let posts = posts.into_iter().enumerate().map(|(i, mut post)| {
        post.id = i as i32 + 1;
        post
    });

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

    // Create the articles table if it doesn't exist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            rss_url TEXT NOT NULL,
            title TEXT NOT NULL,
            link TEXT NOT NULL UNIQUE,
            published TEXT,
            description TEXT
        )",
        [],
    )?;

    // Create the chunks table if it doesn't exist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS article_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            FOREIGN KEY(article_id) REFERENCES articles(id)
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

// Function to process a single post: split into chunks, fetch embeddings, and store in target DB
async fn process_post(
    client: &Client,
    target_db_path: &str,
    post: Post,
) -> Result<(), Box<dyn std::error::Error>> {
    // Extract the description as the content to chunk
    let content = match &post.description {
        Some(desc) => desc.clone(),
        None => {
            return Err("Post does not have a description".into());
        }
    };

    // Insert the original article into the database
    insert_article(target_db_path, &post)?;

    // Split the content into chunks of 200 characters or less
    let chunks = split_into_chunks(&content, 200);

    // Process each chunk: fetch embedding and insert into the target DB
    for (i, chunk_text) in chunks.into_iter().enumerate() {
        // Fetch embedding from Ollama's API
        let embedding = match get_embedding(client, &chunk_text).await {
            Ok(embed) => embed,
            Err(e) => {
                return Err(format!("Failed to get embedding: {}", e).into());
            }
        };

        // Serialize the embedding as JSON
        let embedding_json = serde_json::to_string(&embedding)?;

        // Insert the chunk into the database
        insert_chunk(target_db_path, post.id, &chunk_text, &embedding_json).await?;
    }

    Ok(())
}

// Function to insert an article into the articles table
fn insert_article(db_path: &str, post: &Post) -> SqlResult<()> {
    let conn = Connection::open(db_path)?;

    conn.execute(
        "INSERT OR IGNORE INTO articles (id, rss_url, title, link, published, description) 
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            post.id,
            post.rss_url,
            post.title,
            post.link,
            post.published,
            post.description,
        ],
    )?;
    Ok(())
}

// Function to insert a chunk into the chunks table
async fn insert_chunk(
    db_path: &str,
    article_id: i32,
    text: &str,
    embedding_json: &str,
) -> SqlResult<()> {
    let conn = Connection::open(db_path)?;

    conn.execute(
        "INSERT INTO article_chunks (article_id, text, embedding) 
         VALUES (?1, ?2, ?3)",
        params![article_id, text, embedding_json],
    )?;
    Ok(())
}

// Function to split text into chunks of specified size
fn split_into_chunks(text: &str, chunk_size: usize) -> Vec<String> {
    text.chars()
        .collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

// Function to get embedding from Ollama's API
async fn get_embedding(
    client: &Client,
    text: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let api_url = "http://localhost:11434/v1/embeddings";

    let payload = json!({
        "model": "mxbai-embed-large",
        "input": text,
    });

    let response = client.post(api_url).json(&payload).send().await?;

    if !response.status().is_success() {
        return Err(format!("Ollama API returned status: {}", response.status()).into());
    }

    let response_json: serde_json::Value = response.json().await?;
    let response_json = response_json.get("data").ok_or("Invalid response format")?;
    let response_json = response_json.get(0).ok_or("Invalid response format")?;

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
