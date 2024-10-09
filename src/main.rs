// src/main.rs

use env_logger;
use futures::stream::{self, StreamExt};
use log::{error, info};
use reqwest::Client;
use scraper::{Html, Selector};
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use std::fs::File;
use std::io::BufReader;
use url::Url;

// Define a struct for the RSS feed results
#[derive(Debug, Serialize)]
struct RssFeed {
    blog_url: String,
    rss_url: String,
}

#[tokio::main]
async fn main() {
    // Initialize the logger
    env_logger::init();

    // Load blogs from the configuration file
    let blogs = load_blogs("assets/blogs.json");

    // Initialize the HTTP client
    let client = Client::builder()
        .user_agent("BlogRSSChecker/1.0")
        .build()
        .expect("Failed to build HTTP client");

    // Define the maximum number of concurrent tasks
    let max_concurrency = num_cpus::get();

    // Process blogs concurrently with a limit on concurrency
    let rss_feeds: Vec<RssFeed> = stream::iter(blogs)
        .map(|blog_url| {
            let client = client.clone();
            async move {
                match process_blog(&client, &blog_url).await {
                    Some(rss_url) => Some(RssFeed { blog_url, rss_url }),
                    None => None,
                }
            }
        })
        .buffer_unordered(max_concurrency)
        .filter_map(|x| async move { x })
        .collect()
        .await;

    // Serialize the results to JSON
    let json_output =
        serde_json::to_string_pretty(&rss_feeds).expect("Failed to serialize RSS feeds to JSON");

    // Write the JSON to a file
    match std::fs::write("found_rss.json", json_output) {
        Ok(_) => info!("Successfully wrote RSS feeds to found_rss.json"),
        Err(e) => error!("Failed to write JSON file: {}", e),
    }

    info!("RSS discovery complete.");
}

// Function to load blogs from a JSON file
fn load_blogs(filename: &str) -> Vec<String> {
    let file = File::open(filename).expect("Unable to open blogs.json");
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).expect("JSON was not well-formatted")
}

// Function to process a single blog and find its RSS feed
async fn process_blog(client: &Client, blog_url: &str) -> Option<String> {
    info!("Checking blog: {}", blog_url);

    // Attempt to discover RSS feed via common URLs
    let common_paths = vec!["feed.xml", "rss.xml", "feeds/posts/default?alt=rss"];

    for path in &common_paths {
        let rss_url = format!("{}{}", blog_url, path);
        if let Some(found_url) = check_rss_feed(client, &rss_url).await {
            info!("RSS feed found: {}", found_url);
            return Some(found_url);
        }
    }

    // Attempt to discover RSS feed by parsing HTML
    match discover_rss_by_html(client, blog_url).await {
        Some(rss_url) => {
            info!("RSS feed found via HTML: {}", rss_url);
            Some(rss_url)
        }
        None => {
            info!("No RSS feed found for blog: {}", blog_url);
            None
        }
    }
}

// Function to check if a given URL is a valid RSS feed
async fn check_rss_feed(client: &Client, rss_url: &str) -> Option<String> {
    match client.get(rss_url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                // Simple check: RSS feeds usually start with <?xml
                if let Ok(text) = response.text().await {
                    if text.trim_start().starts_with("<?xml") {
                        return Some(rss_url.to_string());
                    }
                }
            }
            None
        }
        Err(e) => {
            // Optionally log the error
            info!("Failed to fetch {}: {}", rss_url, e);
            None
        }
    }
}

// Function to discover RSS feed by parsing the blog's HTML
async fn discover_rss_by_html(client: &Client, blog_url: &str) -> Option<String> {
    match client.get(blog_url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(html) = response.text().await {
                    let document = Html::parse_document(&html);
                    let selector = Selector::parse("link[rel='alternate']").unwrap();

                    for element in document.select(&selector) {
                        if let Some(mime_type) = element.value().attr("type") {
                            if mime_type == "application/rss+xml"
                                || mime_type == "application/atom+xml"
                            {
                                if let Some(href) = element.value().attr("href") {
                                    // Resolve relative URLs
                                    if let Ok(resolved_url) =
                                        Url::parse(blog_url).and_then(|base| base.join(href))
                                    {
                                        return Some(resolved_url.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            None
        }
        Err(e) => {
            // Optionally log the error
            info!("Failed to fetch HTML from {}: {}", blog_url, e);
            None
        }
    }
}
