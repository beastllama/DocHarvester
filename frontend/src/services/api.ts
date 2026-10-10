import axios, { AxiosInstance } from 'axios'
import { useAuthStore } from '../stores/authStore'

// Types
export interface Project {
  id: number
  name: string
  description?: string
  tags: string[]
  owners: string[]
  created_at: string
  updated_at: string
  document_count: number
  coverage_percentage: number
}

export interface Document {
  id: number
  project_id: number
  doc_id: string
  title: string
  source_type?: string
  source_url?: string
  source_meta: Record<string, any>
  file_type?: string
  last_modified?: string
  created_at: string
  updated_at: string
  chunk_count: number
}

export interface DocumentChunk {
  id: number
  document_id: number
  chunk_index: number
  text: string
  lens_type: string
  confidence_score?: number
  importance_score: number
  is_generated: boolean
  generation_status: string
  tokens?: number
  chunk_metadata: Record<string, any>
}

export interface CoverageRequirement {
  id: number
  project_id: number
  lens_type: string
  is_required: boolean
  min_documents: number
}

export interface CoverageStatus {
  id: number
  project_id: number
  lens_type: string
  status: string
  document_count: number
  chunk_count: number
  coverage_percentage: number
  missing_topics: string[]
  last_checked: string
}

export interface ProjectCoverage {
  project_id: number
  project_name: string
  overall_coverage: number
  requirements: CoverageRequirement[]
  status: CoverageStatus[]
  recommendations: Array<{
    lens_type: string
    priority: string
    action: string
    message: string
    suggested_topics: string[]
  }>
}

export interface ProjectStats {
  total_documents: number
  total_chunks: number
  documents_by_type: Record<string, number>
  coverage_by_lens: Record<string, number>
  recent_activity: Array<{
    action: string
    document: string
    time: string
    type: string
  }>
}

// API Client
class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: '/api/v1',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Add auth interceptor
    this.client.interceptors.request.use((config) => {
      const token = useAuthStore.getState().token
      if (token) {
        config.headers.Authorization = `Bearer ${token}`
      }
      return config
    })

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          useAuthStore.getState().logout()
        }
        return Promise.reject(error)
      }
    )
  }

  // Projects
  async getProjects(skip = 0, limit = 100): Promise<Project[]> {
    const response = await this.client.get('/projects/', { params: { skip, limit } })
    return response.data
  }

  async getProject(id: number): Promise<Project> {
    const response = await this.client.get(`/projects/${id}`)
    return response.data
  }

  async createProject(data: {
    name: string
    description?: string
    tags?: string[]
    owners?: string[]
  }): Promise<Project> {
    const response = await this.client.post('/projects/', data)
    return response.data
  }

  async updateProject(id: number, data: Partial<Project>): Promise<Project> {
    const response = await this.client.put(`/projects/${id}`, data)
    return response.data
  }

  async deleteProject(id: number): Promise<void> {
    await this.client.delete(`/projects/${id}`)
  }

  async getProjectStats(id: number): Promise<ProjectStats> {
    const response = await this.client.get(`/projects/${id}/stats`)
    return response.data
  }

  async startIngestion(id: number): Promise<any> {
    const response = await this.client.post(`/projects/${id}/ingest`)
    return response.data
  }

  async getIngestionStatus(id: number): Promise<any> {
    const response = await this.client.get(`/projects/${id}/ingestion-status`)
    return response.data
  }

  async uploadDocuments(id: number, files: File[]): Promise<any> {
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })
    
    const response = await this.client.post(`/projects/${id}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }

  // Documents
  async searchDocuments(params: {
    q?: string
    project_id?: number
    lens_type?: string
    file_type?: string
    is_generated?: boolean
    page?: number
    limit?: number
  }): Promise<{
    documents: Document[]
    total: number
    page: number
    pages: number
  }> {
    const response = await this.client.get('/documents/', { params })
    return response.data
  }

  async getDocument(id: number): Promise<Document> {
    const response = await this.client.get(`/documents/${id}`)
    return response.data
  }

  async getDocumentContent(id: number): Promise<{
    id: number
    title: string
    file_type: string
    raw_text: string
    source_meta: Record<string, any>
    created_at: string
    last_modified?: string
  }> {
    const response = await this.client.get(`/documents/${id}/content`)
    return response.data
  }

  async getDocumentChunks(id: number, lens_type?: string): Promise<DocumentChunk[]> {
    const response = await this.client.get(`/documents/${id}/chunks`, {
      params: { lens_type },
    })
    return response.data
  }

  async semanticSearch(query: string, project_id?: number, lens_type?: string, limit = 10): Promise<any> {
    const response = await this.client.post('/documents/search/semantic', {
      query,
      project_id,
      lens_type,
      limit,
    })
    return response.data
  }

  async deleteDocument(id: number): Promise<void> {
    await this.client.delete(`/documents/${id}`)
  }

  async reclassifyDocument(id: number): Promise<any> {
    const response = await this.client.put(`/documents/${id}/reclassify`)
    return response.data
  }

  async getLensStatistics(project_id?: number): Promise<any> {
    const response = await this.client.get('/documents/stats/by-lens', {
      params: { project_id },
    })
    return response.data
  }

  // Coverage
  async getProjectRequirements(project_id: number): Promise<CoverageRequirement[]> {
    const response = await this.client.get(`/coverage/requirements/${project_id}`)
    return response.data
  }

  async updateRequirement(
    project_id: number,
    lens_type: string,
    data: { is_required?: boolean; min_documents?: number }
  ): Promise<CoverageRequirement> {
    const response = await this.client.put(`/coverage/requirements/${project_id}/${lens_type}`, data)
    return response.data
  }

  async getCoverageStatus(project_id: number): Promise<ProjectCoverage> {
    const response = await this.client.get(`/coverage/status/${project_id}`)
    return response.data
  }

  async triggerCoverageCheck(project_id: number): Promise<any> {
    const response = await this.client.post(`/coverage/check/${project_id}`)
    return response.data
  }

  async generateMissingDocs(
    project_id: number,
    lens_types?: string[],
    force = false
  ): Promise<any> {
    const response = await this.client.post(`/coverage/generate/${project_id}`, {
      lens_types,
      force,
    })
    return response.data
  }

  async getCoverageGaps(project_id: number): Promise<any> {
    const response = await this.client.get(`/coverage/gaps/${project_id}`)
    return response.data
  }

  // Knowledge Graph
  async getKnowledgeGraphStats(project_id: number): Promise<any> {
    const response = await this.client.get(`/knowledge-graph/projects/${project_id}/stats`)
    return response.data
  }

  async extractEntitiesForProject(project_id: number, options?: { force_reprocess?: boolean, lens_types?: string[] }): Promise<any> {
    const response = await this.client.post(`/knowledge-graph/projects/${project_id}/extract-entities`, options || {})
    return response.data
  }

  async searchEntities(project_id: number, query?: string, entity_types?: string[], limit?: number): Promise<any> {
    const params = new URLSearchParams()
    if (query) params.append('query', query)
    if (entity_types?.length) params.append('entity_types', entity_types.join(','))
    if (limit) params.append('limit', limit.toString())
    
    const response = await this.client.get(`/knowledge-graph/projects/${project_id}/entities?${params}`)
    return response.data
  }

  async checkNeo4jStatus(project_id: number): Promise<any> {
    const response = await this.client.get(`/knowledge-graph/projects/${project_id}/neo4j-status`)
    return response.data
  }

  async refreshKnowledgeGraph(project_id: number): Promise<any> {
    const response = await this.client.post(`/knowledge-graph/projects/${project_id}/refresh-knowledge-graph`)
    return response.data
  }

  // Connectors
  async getConnectors(): Promise<any> {
    const response = await this.client.get('/connectors/')
    return response.data
  }

  async getConnectorConfig(project_id: number, connector_type: string): Promise<any> {
    const response = await this.client.get(`/connectors/${project_id}/${connector_type}`)
    return response.data
  }

  async updateConnectorConfig(
    project_id: number,
    connector_type: string,
    config: any
  ): Promise<any> {
    const response = await this.client.post(`/connectors/project/${project_id}/configure`, {
      connector_type,
      config
    })
    return response.data
  }

  async testConnector(project_id: number, connector_type: string): Promise<any> {
    const response = await this.client.post(`/connectors/${project_id}/${connector_type}/test`)
    return response.data
  }

  async getProjectConnectors(project_id: number): Promise<any> {
    const response = await this.client.get(`/connectors/project/${project_id}/configurations`)
    return response.data
  }

  // Personal API tokens (for Hermes Agent and other MCP clients)
  async getApiTokens(): Promise<ApiToken[]> {
    const response = await this.client.get('/auth/tokens')
    return response.data
  }

  async createApiToken(name: string): Promise<ApiToken & { token: string }> {
    const response = await this.client.post('/auth/tokens', { name })
    return response.data
  }

  async revokeApiToken(id: number): Promise<void> {
    await this.client.delete(`/auth/tokens/${id}`)
  }
}

export interface ApiToken {
  id: number
  name: string
  created_at: string
  last_used_at?: string | null
  revoked_at?: string | null
}

// Export singleton instance
export const api = new ApiClient() 