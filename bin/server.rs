// smart_qa_server/src/main.rs

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    error::Error,
    sync::{Arc, Mutex},
};
use tokio::sync::Semaphore;
use warp::Filter;

use rusqlite::{params, Connection, Result as SqlResult};

use log::{error, info};

#[derive(Clone)]
struct AppState {
    db_conn: Arc<Mutex<Connection>>,
    client: Arc<reqwest::Client>,
    semaphore: Arc<Semaphore>,
}

#[derive(Debug, Deserialize)]
struct Question {
    question: String,
}

#[derive(Debug, Serialize)]
struct AnswerResponse {
    answer: String,
    sources: Vec<Source>,
}

#[derive(Debug, Serialize)]
struct Source {
    title: String,
    link: String,
    description: Option<String>,
}

#[tokio::main]
async fn main() {
    // Initialize the logger
    env_logger::init();

    // Initialize the SQLite connection
    let conn = match Connection::open("assets/embedded_posts.db") {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to connect to database: {}", e);
            return;
        }
    };

    // Wrap the connection in Arc and Mutex for thread-safe access
    let db_conn = Arc::new(Mutex::new(conn));

    // Initialize the HTTP client
    let client = Arc::new(
        reqwest::Client::builder()
            .user_agent("SmartQA_Server/1.0")
            .build()
            .expect("Failed to build HTTP client"),
    );

    // Define rate limiting: 5 requests per second
    let rate_limit = 5;
    let semaphore = Arc::new(Semaphore::new(rate_limit));

    // Create AppState
    let state = AppState {
        db_conn: Arc::clone(&db_conn),
        client: Arc::clone(&client),
        semaphore: Arc::clone(&semaphore),
    };

    // Define the /ask route
    let ask_route = warp::path("ask")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_state(state.clone()))
        .and_then(handle_query);

    // Define the index route
    let index_route = warp::path::end().and(warp::get()).and_then(serve_index);

    // Combine routes
    let routes = ask_route.or(index_route);

    info!("Starting server on http://127.0.0.1:8080");

    // Start the server
    warp::serve(routes).run(([127, 0, 0, 1], 8080)).await;
}

// Function to pass AppState to handlers
fn with_state(
    state: AppState,
) -> impl Filter<Extract = (AppState,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || state.clone())
}

// Handler to serve the embedded index.html
async fn serve_index() -> Result<impl warp::Reply, warp::Rejection> {
    Ok(warp::reply::html(INDEX_HTML))
}

// Handler for the /ask endpoint
async fn handle_query(
    question: Question,
    state: AppState,
) -> Result<impl warp::Reply, warp::Rejection> {
    let question_text = question.question.trim().to_string();
    if question_text.is_empty() {
        return Ok(warp::reply::with_status(
            warp::reply::json(&json!({"error": "Question cannot be empty"})),
            warp::http::StatusCode::BAD_REQUEST,
        ));
    }

    // Acquire a semaphore permit for rate limiting
    let permit = state.semaphore.acquire().await;
    if permit.is_err() {
        return Ok(warp::reply::with_status(
            warp::reply::json(&json!({"error": "Rate limit exceeded"})),
            warp::http::StatusCode::TOO_MANY_REQUESTS,
        ));
    }

    // Clone necessary parts of the state
    let db_conn = Arc::clone(&state.db_conn);
    let client = Arc::clone(&state.client);

    // Process the query
    match process_query(db_conn, client, question_text).await {
        Ok(response) => Ok(warp::reply::with_status(
            warp::reply::json(&response),
            warp::http::StatusCode::OK,
        )),
        Err(e) => {
            error!("Error processing query: {}", e);
            Ok(warp::reply::with_status(
                warp::reply::json(&json!({"error": "Failed to process the query"})),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }
}

// Function to process the query
async fn process_query(
    db_conn: Arc<Mutex<Connection>>,
    client: Arc<reqwest::Client>,
    question: String,
) -> Result<AnswerResponse, Box<dyn std::error::Error>> {
    // Generate embedding for the question
    let question_embedding = generate_embedding(&question).await?;

    // Fetch relevant sources based on embeddings
    let sources = fetch_relevant_sources(&db_conn, &question_embedding)?;

    if sources.is_empty() {
        return Ok(AnswerResponse {
            answer: "No relevant sources found.".to_string(),
            sources: Vec::new(),
        });
    }

    // Construct context from sources
    let context = sources
        .iter()
        .map(|s| {
            format!(
                "[{}]({}): {}",
                s.title,
                s.link,
                s.description.as_deref().unwrap_or("")
            )
        })
        .collect::<Vec<String>>()
        .join("\n");

    // Generate the prompt for the LLM
    let prompt = format!(
        "You are an AI assistant. Use the following context to answer the question.\n\nContext:\n{}\n\nQuestion: {}\n\nAnswer:",
        context, question
    );

    // Get the answer from the LLM
    let answer = get_llm_answer(&client, &prompt).await?;

    Ok(AnswerResponse { answer, sources })
}

// Function to generate embedding for a given text
async fn generate_embedding(text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    // Define the Ollama API endpoint for generating embeddings
    let api_url = "http://localhost:11434/v1/embeddings"; // Adjust the URL based on your Ollama setup

    // Create the request payload
    let payload = json!({
        "input": text,
        "model": "mxbai-embed-large" // Adjust based on the available models in Ollama
    });

    // Send the POST request
    let client = Client::new();
    let response = client.post(api_url).json(&payload).send().await?;

    // Check if the response status is successful
    if !response.status().is_success() {
        return Err(format!("Embedding API returned status: {}", response.status()).into());
    }

    // Parse the response JSON
    let response_json: serde_json::Value = response.json().await?;

    // Extract the embedding from the response
    let response_json = response_json.get("data").ok_or("Invalid response format")?;
    let response_json = response_json.get(0).ok_or("Invalid response format")?;
    if let Some(embedding) = response_json.get("embedding").and_then(|e| e.as_array()) {
        // Convert the embedding array to Vec<f32>
        let embedding_vec: Vec<f32> = embedding
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        // Return the embedding vector
        Ok(embedding_vec)
    } else {
        Err("Invalid embedding format in the API response".into())
    }
}

// Function to fetch relevant sources based on embeddings similarity
fn fetch_relevant_sources(
    db_conn: &Arc<Mutex<Connection>>,
    question_embedding: &[f32],
) -> Result<Vec<Source>, rusqlite::Error> {
    let conn = db_conn.lock().unwrap();

    // Fetch all embeddings from the database
    let mut stmt =
        conn.prepare("SELECT title, link, description, embedding FROM embedded_posts")?;
    let source_iter = stmt.query_map([], |row| {
        let title: String = row.get(0)?;
        let link: String = row.get(1)?;
        let description: Option<String> = row.get(2)?;
        let embedding_str: String = row.get(3)?;
        let embedding: Vec<f32> = serde_json::from_str(&embedding_str).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(2, rusqlite::types::Type::Text, Box::new(e))
        })?;
        Ok((title, link, description, embedding))
    })?;

    let mut sources_with_similarity: Vec<(Source, f32)> = Vec::new();

    for source in source_iter {
        let (title, link, description, embedding) = source?;
        let similarity = cosine_similarity(question_embedding, &embedding);
        sources_with_similarity.push((
            Source {
                title,
                link,
                description,
            },
            similarity,
        ));
    }

    // Sort sources by similarity in descending order
    sources_with_similarity.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Select top 5 sources with similarity above a threshold (e.g., 0.5)
    let top_sources = sources_with_similarity
        .into_iter()
        .filter(|&(_, sim)| sim > 0.5)
        .take(5)
        .map(|(source, _)| source)
        .collect();

    Ok(top_sources)
}

// Function to compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        0.0
    } else {
        dot_product / (magnitude_a * magnitude_b)
    }
}

// Function to get LLM answer from Ollama's API
async fn get_llm_answer(
    client: &reqwest::Client,
    prompt: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    // Define the Ollama API endpoint for generating completions
    let api_url = "http://localhost:11434/v1/completions"; // Adjust based on Ollama's API

    // Create the request payload
    let payload = json!({
        "model": "llama3.1:8b", // Adjust based on the available models in Ollama
        "prompt": prompt,
        "temperature": 1.0,
        "options": {
            "num_ctx": 32 * 1024,
        }
    });

    // Send the POST request
    let response = client.post(api_url).json(&payload).send().await?;

    if !response.status().is_success() {
        return Err(format!("Ollama API returned status: {}", response.status()).into());
    }

    // Parse the response JSON
    let response_json: serde_json::Value = response.json().await?;

    // Extract the answer
    if let Some(answer) = response_json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
    {
        Ok(answer.trim().to_string())
    } else {
        Err("Invalid response format from LLM".into())
    }
}

// Embedded index.html content
const INDEX_HTML: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Smart Q&A</title>
    <style>
        /* Embedded CSS */
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }

        .container {
            width: 60%;
            margin: 50px auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        h1 {
            text-align: center;
        }

        textarea {
            width: 100%;
            height: 100px;
            padding: 10px;
            margin-top: 20px;
            border-radius: 4px;
            border: 1px solid #ccc;
            resize: vertical;
        }

        button {
            display: block;
            width: 100%;
            padding: 10px;
            margin-top: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }

        button:hover {
            background-color: #0056b3;
        }

        #answer {
            margin-top: 20px;
            padding: 15px;
            background-color: #e9ecef;
            border-radius: 4px;
            min-height: 100px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Smart Q&A</h1>
        <textarea id="question" placeholder="Type your question here..."></textarea>
        <button id="ask-button">Ask</button>
        <div id="answer"></div>
    </div>
    <script>
        // Embedded JavaScript
        document.getElementById('ask-button').addEventListener('click', async () => {
            const question = document.getElementById('question').value.trim();
            const answerDiv = document.getElementById('answer');

            if (!question) {
                answerDiv.innerHTML = '<p>Please enter a question.</p>';
                return;
            }

            answerDiv.innerHTML = '<p>Loading...</p>';

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question })
                });

                if (!response.ok) {
                    throw new Error(`Error: ${response.statusText}`);
                }

                const data = await response.json();
                const answerText = data.answer || 'No answer found.';
                const answer = (new showdown.Converter()).makeHtml(answerText);

                answerDiv.innerHTML = `
                    <h3>Answer:</h3>
                    <p>${answer}</p>
                    <h4>Sources:</h4>
                    <ul>
                        ${data.sources.map(source => `<li><a href="${source.link}" target="_blank">${source.title}</a></li>`).join('')}
                    </ul>
                `;
            } catch (error) {
                console.error(error);
                answerDiv.innerHTML = `<p>Failed to get answer. Please try again later.</p>`;
            }
        });
    </script>
    <script src=" https://cdn.jsdelivr.net/npm/showdown@1.9.1/dist/showdown.min.js"></script>
</body>
</html>
"#;
