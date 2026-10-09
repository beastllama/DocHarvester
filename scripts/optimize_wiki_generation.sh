#!/bin/bash

echo "🚀 Optimizing DocHarvester for Wiki Generation & Knowledge Graph"
echo "================================================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_status "Docker is running"

# Step 1: Stop existing services
echo ""
echo "🛑 Stopping existing services..."
docker-compose down

# Step 2: Backup current environment (if exists)
if [ -f ".env" ]; then
    echo "📋 Backing up current .env to .env.backup"
    cp .env .env.backup
fi

# Step 3: Apply optimized configuration
echo ""
echo "⚙️ Applying optimized configuration..."
cp env.optimized .env
print_status "Environment configuration updated"

# Step 4: Use optimized docker-compose
echo ""
echo "🐳 Using optimized Docker Compose configuration..."
if [ -f "docker-compose.yml" ]; then
    cp docker-compose.yml docker-compose.backup.yml
    print_status "Backed up existing docker-compose.yml"
fi

cp docker-compose.optimized.yml docker-compose.yml
print_status "Docker Compose configuration updated"

# Step 5: Clean up Docker resources
echo ""
echo "🧹 Cleaning up Docker resources..."
docker system prune -f
print_status "Docker cleanup completed"

# Step 6: Rebuild and start services
echo ""
echo "🏗️ Building and starting optimized services..."
docker-compose build --no-cache

echo ""
echo "🚀 Starting services with optimized configuration..."
docker-compose up -d

# Step 7: Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to become healthy..."

wait_for_service() {
    local service=$1
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for $service..."
    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps $service | grep -q "Up (healthy)"; then
            print_status "$service is healthy"
            return 0
        fi
        
        if [ $((attempt % 5)) -eq 0 ]; then
            echo "Still waiting for $service (attempt $attempt/$max_attempts)..."
        fi
        
        sleep 10
        attempt=$((attempt + 1))
    done
    
    print_warning "$service took longer than expected to become healthy"
    return 1
}

# Wait for core services
wait_for_service "postgres"
wait_for_service "redis"
wait_for_service "ollama"

# Step 8: Setup Ollama models
echo ""
echo "📥 Setting up Ollama models..."

# Wait a bit more for Ollama to be fully ready
sleep 10

# Check if gemma:2b is available, pull if needed
echo "🔍 Checking Ollama models..."
if docker exec docharvester-ollama ollama list | grep -q "gemma:2b"; then
    print_status "gemma:2b model is already available"
else
    echo "🔽 Pulling gemma:2b model (this may take a few minutes)..."
    if docker exec docharvester-ollama ollama pull gemma:2b; then
        print_status "gemma:2b model pulled successfully"
    else
        print_error "Failed to pull gemma:2b model"
    fi
fi

# Step 9: Wait for application services
wait_for_service "backend"

# Step 10: Test the optimization
echo ""
echo "🧪 Testing optimized configuration..."

# Test Ollama connection
echo "Testing Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    print_status "Ollama API is responding"
else
    print_warning "Ollama API not responding - may need more time to start"
fi

# Test backend health
echo "Testing backend..."
sleep 5
if curl -s http://localhost:8000/health > /dev/null; then
    print_status "Backend API is responding"
else
    print_warning "Backend API not responding - may need more time to start"
fi

# Step 11: Show service status
echo ""
echo "📊 Service Status:"
echo "=================="
docker-compose ps

# Step 12: Show optimization summary
echo ""
echo "🎯 Optimization Summary:"
echo "======================="
echo "✅ LLM Configuration Fixed:"
echo "   - Provider: LOCAL (gemma:2b)"
echo "   - Timeout: 90 seconds (reduced from 300s)"
echo "   - Response caching enabled"
echo ""
echo "✅ Resource Allocation Optimized:"
echo "   - Ollama: 8 CPU cores, 12GB RAM"
echo "   - Backend: 3 CPU cores, 6GB RAM"
echo "   - Workers: Limited concurrency (2 tasks)"
echo ""
echo "✅ Knowledge Graph Integration:"
echo "   - Entity extraction optimized"
echo "   - Smart chunking enabled"
echo "   - Context window management"
echo ""
echo "✅ Wiki Generation Improvements:"
echo "   - Progressive generation with caching"
echo "   - Knowledge graph context integration"
echo "   - Timeout handling and fallbacks"

# Step 13: Usage instructions
echo ""
echo "🚀 Next Steps:"
echo "============="
echo "1. Access the application:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo "   - Neo4j Browser: http://localhost:7474"
echo ""
echo "2. Test wiki generation:"
echo "   - Upload documents to a project"
echo "   - Wait for entity extraction to complete"
echo "   - Generate wiki from the project settings"
echo ""
echo "3. Monitor performance:"
echo "   - Check logs: docker-compose logs -f"
echo "   - Monitor resources: docker stats"
echo ""
echo "4. Troubleshooting:"
echo "   - If timeouts persist, check: docker-compose logs ollama"
echo "   - For errors, check: docker-compose logs backend"
echo "   - Reset if needed: ./scripts/reset_optimized.sh"

# Step 14: Create reset script
echo ""
echo "📝 Creating reset script..."
cat > scripts/reset_optimized.sh << 'EOF'
#!/bin/bash
echo "🔄 Resetting to original configuration..."

# Stop services
docker-compose down

# Restore backups if they exist
if [ -f ".env.backup" ]; then
    cp .env.backup .env
    echo "✅ Restored original .env"
fi

if [ -f "docker-compose.backup.yml" ]; then
    cp docker-compose.backup.yml docker-compose.yml
    echo "✅ Restored original docker-compose.yml"
fi

# Start with original configuration
docker-compose up -d

echo "✅ Reset complete. Original configuration restored."
EOF

chmod +x scripts/reset_optimized.sh
print_status "Reset script created at scripts/reset_optimized.sh"

echo ""
echo "🎉 Optimization Complete!"
echo "========================"
print_status "DocHarvester has been optimized for wiki generation and knowledge graph integration"
print_warning "Please test the system and monitor for any issues"
echo "📖 Full documentation: README.md" 