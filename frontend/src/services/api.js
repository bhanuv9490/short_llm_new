import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || '';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  timeout: 30000, // 30 seconds timeout
});

// Request interceptor for API calls
api.interceptors.request.use(
  (config) => {
    // You can add auth headers here if needed
    // const token = localStorage.getItem('token');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for API calls
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle common errors
    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout. Please try again.');
    }
    
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      const { status, data } = error.response;
      
      if (status === 401) {
        // Handle unauthorized
        // window.location.href = '/login';
      } else if (status === 404) {
        throw new Error('The requested resource was not found.');
      } else if (status >= 500) {
        throw new Error('Server error. Please try again later.');
      }
      
      // If the server sent a custom error message, use it
      if (data && data.detail) {
        throw new Error(data.detail);
      }
    } else if (error.request) {
      // The request was made but no response was received
      throw new Error('No response from server. Please check your connection.');
    }
    
    throw error;
  }
);

/**
 * Generate text using the specified model
 * @param {string} prompt - The input prompt
 * @param {string} model - The model to use ('phi2' or 'gpt2')
 * @param {Object} params - Additional parameters
 * @param {number} [params.max_length=100] - Maximum length of the generated text
 * @param {number} [params.temperature=0.7] - Controls randomness (0.1-2.0)
 * @param {number} [params.top_p=0.9] - Nucleus sampling parameter (0.0-1.0)
 * @returns {Promise<Object>} - Response object with status and generated text
 */
export const generateText = async (prompt, model, params = {}) => {
  try {
    const response = await api.post('/api/generate', {
      prompt: prompt.trim(),
      model: model || 'phi2',
      max_length: Math.min(Number(params.max_length) || 100, 1000),
      temperature: Math.max(0.1, Math.min(Number(params.temperature) || 0.7, 2.0)),
      top_p: Math.max(0.1, Math.min(Number(params.top_p) || 0.9, 1.0))
    }, {
      timeout: 180000, // 3 minute timeout
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });
    
    if (response.data && response.data.status === 'success') {
      return {
        ...response.data,
        success: true,
        timestamp: new Date().toISOString()
      };
    }
    
    throw new Error(response.data?.detail || 'Invalid response from server');
  } catch (error) {
    console.error('API Error:', error);
    const errorMessage = error.response?.data?.detail || 
                       error.response?.data?.error || 
                       error.message || 
                       'Failed to generate text';
    
    return {
      status: 'error',
      error: errorMessage,
      success: false,
      timestamp: new Date().toISOString()
    };
  }
};

export const healthCheck = async () => {
  try {
    const response = await api.get('/api/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw new Error(error.message || 'Failed to connect to the server');
  }
};

export default {
  generateText,
  healthCheck,
  api, // Export the axios instance for other API calls if needed
};
