import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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

export const generateText = async (prompt, model, params = {}) => {
  try {
    const response = await api.post('/api/generate', {
      prompt,
      model,
      ...params,
    });
    
    if (response.data && response.data.status === 'success') {
      return response.data;
    }
    
    throw new Error('Invalid response from server');
  } catch (error) {
    console.error('API Error:', error);
    throw error.response?.data || { error: error.message || 'Failed to generate text' };
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
