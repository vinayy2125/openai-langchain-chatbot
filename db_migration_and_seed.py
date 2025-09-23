import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def get_connection(dbname=DB_NAME):
    return psycopg2.connect(
        dbname=dbname,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        options="-c client_encoding=UTF8"
    )

# ----------------------------
# Drop existing tables
# ----------------------------
def drop_tables():
    conn = get_connection()
    cursor = conn.cursor()
    print("⚠️ Dropping existing tables if they exist...")
    cursor.execute("DROP TABLE IF EXISTS messages CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS sessions CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS users CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS prompts CASCADE;")
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Old tables dropped successfully.")

# ----------------------------
# Create updated tables with constraints
# ----------------------------
def create_tables():
    conn = get_connection()
    cursor = conn.cursor()
    print("🛠 Creating tables...")

    # Enable UUID extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")

    # Prompts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prompts (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        prompt_text TEXT NOT NULL,
        parent_id UUID REFERENCES prompts(id) ON DELETE CASCADE,
        response_text TEXT,
        display_order INT DEFAULT 0,
        type VARCHAR(50) DEFAULT 'root',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_prompt_text_per_parent UNIQUE (parent_id, prompt_text)
    );
    """)

    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        username VARCHAR(255),
        email VARCHAR(255) UNIQUE,
        mobile VARCHAR(50),
        browser VARCHAR(255),
        ip VARCHAR(50),
        first_name VARCHAR(255),
        last_name VARCHAR(255),
        email_opt_in BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Sessions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
        session_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
        title VARCHAR(255),
        browser VARCHAR(255),
        ip VARCHAR(50),
        is_active BOOLEAN DEFAULT TRUE,
        current_prompt_id UUID REFERENCES prompts(id) ON DELETE SET NULL,
        requirements_met BOOLEAN DEFAULT FALSE,
        follow_up_count INT DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_interaction_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Create partial unique index: only one active session per user
    cursor.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS unique_active_session_per_user
    ON sessions(user_id)
    WHERE is_active;
    """)

    # Messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE NOT NULL,
        content TEXT NOT NULL,
        role VARCHAR(50) NOT NULL,
        reply_to UUID REFERENCES messages(id) ON DELETE SET NULL,
        follow_up_to UUID REFERENCES messages(id) ON DELETE SET NULL,
        follow_up_depth INT DEFAULT 0,
        metadata JSONB DEFAULT '{}'::jsonb,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Tables created successfully.")

# ----------------------------
# Seed initial prompts
# ----------------------------
def seed_prompts():
    conn = get_connection()
    cursor = conn.cursor()
    print("🌱 Seeding initial prompts...")

    # Root prompts - Technology Stack Categories
    root_prompts = [
        ("Front-end Development", "We specialize in modern front-end technologies including AngularJS, ReactJS, Vue.js, JavaScript, CSS3, and HTML5.", 1),
        ("Back-end Development", "Our back-end expertise covers .NET, Node.js, and PHP for robust server-side solutions.", 2),
        ("Mobile Development", "We develop mobile applications using Android, iOS, Ionic, and React Native technologies.", 3),
        ("Cloud Solutions", "We provide cloud services on AWS, Google Cloud, and Azure platforms.", 4),
        ("Database Solutions", "Our database expertise includes MySQL, MongoDB, and PostgreSQL for your data needs.", 5),
        ("DevOps Services", "We offer DevOps solutions using Azure DevOps, Docker, and Kubernetes for streamlined deployment.", 6)
    ]

    for text, response, order in root_prompts:
        cursor.execute("""
        INSERT INTO prompts (prompt_text, response_text, display_order, type)
        VALUES (%s, %s, %s, 'root')
        RETURNING id;
        """, (text, response, order))

    # Fetch root IDs - store them as we insert
    root_map = {}
    for text, _, _ in root_prompts:
        cursor.execute("SELECT id FROM prompts WHERE prompt_text = %s AND type = 'root';", (text,))
        root_id = cursor.fetchone()[0]
        root_map[text] = root_id

    # Example follow-ups for each root prompt
    followups = {
        "Front-end Development": [
            ("AngularJS Projects", "I can explain our AngularJS development capabilities and experience.", 1),
            ("ReactJS Applications", "Let me share our ReactJS development expertise and portfolio.", 2),
            ("Vue.js Solutions", "We specialize in Vue.js for modern, reactive user interfaces.", 3),
            ("JavaScript & CSS3", "Our team excels in advanced JavaScript and CSS3 implementations.", 4)
        ],
        "Back-end Development": [
            (".NET Solutions", "We develop robust .NET applications for enterprise needs.", 1),
            ("Node.js Development", "Our Node.js expertise covers scalable server-side applications.", 2),
            ("PHP Applications", "We create dynamic PHP solutions for web development.", 3),
            ("API Development", "We can build RESTful APIs using any of our back-end technologies.", 4)
        ],
        "Mobile Development": [
            ("Native Android", "We develop high-performance native Android applications.", 1),
            ("Native iOS", "Our iOS development covers Swift and Objective-C applications.", 2),
            ("Cross-platform Ionic", "Ionic framework for hybrid mobile app development.", 3),
            ("React Native Apps", "React Native for cross-platform mobile solutions.", 4)
        ],
        "Cloud Solutions": [
            ("AWS Services", "We implement comprehensive AWS cloud solutions and migrations.", 1),
            ("Google Cloud Platform", "Our GCP expertise covers compute, storage, and AI services.", 2),
            ("Microsoft Azure", "Azure solutions for enterprise cloud infrastructure.", 3),
            ("Cloud Migration", "We can help migrate your existing systems to the cloud.", 4)
        ],
        "Database Solutions": [
            ("MySQL Databases", "Expert MySQL database design and optimization services.", 1),
            ("MongoDB Solutions", "NoSQL MongoDB development for modern applications.", 2),
            ("PostgreSQL Systems", "Advanced PostgreSQL database solutions and management.", 3),
            ("Database Migration", "We can help migrate between different database systems.", 4)
        ],
        "DevOps Services": [
            ("Azure DevOps", "Complete CI/CD pipeline setup using Azure DevOps.", 1),
            ("Docker Containerization", "Container solutions using Docker for application deployment.", 2),
            ("Kubernetes Orchestration", "Kubernetes for container orchestration and scaling.", 3),
            ("Infrastructure as Code", "Automated infrastructure management and deployment.", 4)
        ]
    }

    for root_text, follow_up_list in followups.items():
        parent_id = root_map[root_text]
        for text, response, order in follow_up_list:
            cursor.execute("""
            INSERT INTO prompts (prompt_text, parent_id, response_text, display_order, type)
            VALUES (%s, %s, %s, %s, 'follow_up')
            """, (text, parent_id, response, order))

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Initial prompts seeded successfully.")

# ----------------------------
# Run migration
# ----------------------------
if __name__ == "__main__":
    drop_tables()
    create_tables()
    seed_prompts()
    print("🎉 Migration and seeding completed. DB is ready for the updated API!")
