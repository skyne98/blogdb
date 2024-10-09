use atom_syndication as atom;
use env_logger;
use futures::stream::{self, StreamExt};
use log::{error, info};
use reqwest::Client;
use rss::Channel;
use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};
use url::Url;

#[derive(Debug, Deserialize)]
struct RssFeed {
    blog_url: String,
    rss_url: String,
}

#[derive(Debug, Serialize)]
struct Post {
    rss_url: String,
    title: String,
    link: String,
    published: Option<String>,
    description: Option<String>,
}

#[tokio::main]
async fn main() {
    // Initialize the logger
    env_logger::init();

    // Load RSS feeds from the configuration file
    let rss_feeds = load_rss_feeds("assets/found_rss.json");

    if rss_feeds.is_empty() {
        info!("No RSS feeds found in found_rss.json.");
        return;
    }

    // Initialize the SQLite database
    let conn = match initialize_database("rss_posts.db") {
        Ok(conn) => Arc::new(conn),
        Err(e) => {
            error!("Failed to initialize the database: {}", e);
            return;
        }
    };

    // Initialize the HTTP client
    let client = Client::builder()
        .user_agent("RSSDownloader/1.0")
        .build()
        .expect("Failed to build HTTP client");

    let client = Arc::new(client);

    // Define rate limiting: e.g., 5 requests per second
    let rate_limit = 5;
    let semaphore = Arc::new(Semaphore::new(rate_limit));

    // Process RSS feeds concurrently with rate limiting
    let tasks = stream::iter(rss_feeds)
        .map(|rss_feed| {
            let client = Arc::clone(&client);
            let conn = Arc::clone(&conn);
            let semaphore = Arc::clone(&semaphore);
            async move {
                let _permit = semaphore.acquire().await.unwrap();
                match fetch_and_store_feed(&client, &conn, &rss_feed).await {
                    Ok(_) => info!("Processed RSS feed: {}", rss_feed.rss_url),
                    Err(e) => error!("Error processing {}: {}", rss_feed.rss_url, e),
                }
                // Sleep to maintain the rate limit
                sleep(Duration::from_secs_f32(1.0 / rate_limit as f32)).await;
            }
        })
        .buffer_unordered(rate_limit);

    tasks.collect::<Vec<()>>().await;

    info!("RSS downloading complete.");
}

// Function to load RSS feeds from a JSON file
fn load_rss_feeds(filename: &str) -> Vec<RssFeed> {
    let file = File::open(filename).expect("Unable to open found_rss.json");
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).expect("JSON was not well-formatted")
}

// Function to initialize the SQLite database
fn initialize_database(db_path: &str) -> SqlResult<Connection> {
    let conn = Connection::open(db_path)?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rss_url TEXT NOT NULL,
            title TEXT NOT NULL,
            link TEXT NOT NULL UNIQUE,
            published TEXT,
            description TEXT
        )",
        [],
    )?;
    Ok(conn)
}

// Function to fetch and store RSS or Atom feed
async fn fetch_and_store_feed(
    client: &Client,
    conn: &Connection,
    rss_feed: &RssFeed,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = client.get(&rss_feed.rss_url).send().await?;

    if !response.status().is_success() {
        return Err(format!(
            "Failed to fetch RSS feed {}: {}",
            rss_feed.rss_url,
            response.status()
        )
        .into());
    }

    let content = response.text().await?;

    // Attempt to parse as RSS
    match Channel::read_from(content.as_bytes()) {
        Ok(channel) => {
            for item in channel.items() {
                let post = Post {
                    rss_url: rss_feed.rss_url.clone(),
                    title: item.title().unwrap_or("No Title").to_string(),
                    link: item.link().unwrap_or("No Link").to_string(),
                    published: item.pub_date().map(|d| d.to_string()),
                    description: item.description().map(|d| d.to_string()),
                };

                // Insert into the database
                let result = conn.execute(
                    "INSERT OR IGNORE INTO posts (rss_url, title, link, published, description) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        post.rss_url,
                        post.title,
                        post.link,
                        post.published,
                        post.description,
                    ],
                );

                match result {
                    Ok(_) => (),
                    Err(e) => error!("Failed to insert post {}: {}", post.link, e),
                }
            }
            Ok(())
        }
        Err(_) => {
            // If RSS parsing fails, attempt to parse as Atom
            match atom::Feed::read_from(content.as_bytes()) {
                Ok(atom_feed) => {
                    for entry in atom_feed.entries() {
                        let post = Post {
                            rss_url: rss_feed.rss_url.clone(),
                            title: entry.title().to_string(),
                            link: entry
                                .links()
                                .iter()
                                .find(|link| link.rel() == "alternate")
                                .and_then(|link| Some(link.href()))
                                .unwrap_or("No Link")
                                .to_string(),
                            published: entry.published().map(|d| d.to_string()),
                            description: entry.content().map(|c| c.value().unwrap().to_string()),
                        };

                        // Insert into the database
                        let result = conn.execute(
                            "INSERT OR IGNORE INTO posts (rss_url, title, link, published, description) VALUES (?1, ?2, ?3, ?4, ?5)",
                            params![
                                post.rss_url,
                                post.title,
                                post.link,
                                post.published,
                                post.description,
                            ],
                        );

                        match result {
                            Ok(_) => (),
                            Err(e) => error!("Failed to insert post {}: {}", post.link, e),
                        }
                    }
                    Ok(())
                }
                Err(e_atom) => {
                    // Both RSS and Atom parsing failed
                    Err(format!(
                        "Failed to parse RSS feed {} as both RSS and Atom: {:?}",
                        rss_feed.rss_url, e_atom
                    )
                    .into())
                }
            }
        }
    }
}
