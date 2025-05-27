import React, { useState, useEffect } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { 
  Container, 
  Box, 
  Paper, 
  TextField, 
  Button, 
  Typography, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Switch, 
  FormControlLabel,
  CircularProgress,
  Alert,
  Snackbar
} from '@mui/material';
import { generateText, healthCheck } from './services/api';

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [model, setModel] = useState('phi2');
  const [maxLength, setMaxLength] = useState(100);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [isLoading, setIsLoading] = useState(false);
  const [isBackendAvailable, setIsBackendAvailable] = useState(true);
  const [error, setError] = useState(null);
  const [snackbarOpen, setSnackbarOpen] = useState(false);

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#556cd6',
      },
      secondary: {
        main: '#19857b',
      },
    },
  });

  // Check backend health on component mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        await healthCheck();
        setIsBackendAvailable(true);
      } catch (err) {
        console.error('Backend not available:', err);
        setIsBackendAvailable(false);
        setError('Backend server is not available. Please make sure the backend is running on port 8000.');
        setSnackbarOpen(true);
      }
    };
    
    checkBackend();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      setSnackbarOpen(true);
      return;
    }

    setIsLoading(true);
    setError('');
    setResponse('');

    try {
      // Show loading state
      setResponse('Loading model and generating response. This may take a minute...');
      
      // Call the API
      const result = await generateText(prompt, model, {
        max_length: maxLength,
        temperature,
        top_p: topP
      });
      
      if (result.success) {
        setResponse(result.response || 'No response generated');
      } else {
        setError(result.error || 'Failed to generate response');
        setSnackbarOpen(true);
      }
    } catch (err) {
      console.error('Error in handleSubmit:', err);
      setError(err.message || 'An unexpected error occurred');
      setSnackbarOpen(true);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbarOpen(false);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" component="h1">
            LLM Playground
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center',
              gap: 1,
              px: 1.5,
              py: 0.5,
              borderRadius: 1,
              bgcolor: isBackendAvailable ? 'success.light' : 'error.light',
              color: 'white',
              fontSize: '0.75rem',
              fontWeight: 'medium'
            }}>
              {isBackendAvailable ? 'Backend Connected' : 'Backend Unavailable'}
            </Box>
            <FormControlLabel
              control={
                <Switch
                  checked={darkMode}
                  onChange={() => setDarkMode(!darkMode)}
                  color="primary"
                />
              }
              label={darkMode ? 'Dark' : 'Light'}
            />
          </Box>
        </Box>

        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <form onSubmit={handleSubmit}>
            <FormControl fullWidth sx={{ mb: 2 }}>
              <TextField
                label="Enter your prompt"
                multiline
                rows={4}
                variant="outlined"
                fullWidth
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                disabled={isLoading}
              />
            </FormControl>

            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
              <FormControl sx={{ minWidth: 120 }}>
                <InputLabel>Model</InputLabel>
                <Select
                  value={model}
                  label="Model"
                  onChange={(e) => setModel(e.target.value)}
                  disabled={isLoading}
                >
                  <MenuItem value="phi2">Phi-2</MenuItem>
                  <MenuItem value="gpt2">GPT-2</MenuItem>
                </Select>
              </FormControl>

              <FormControl sx={{ minWidth: 100 }}>
                <TextField
                  label="Max Length"
                  type="number"
                  value={maxLength}
                  onChange={(e) => setMaxLength(parseInt(e.target.value))}
                  disabled={isLoading}
                  inputProps={{ min: 1, max: 1000 }}
                />
              </FormControl>

              <FormControl sx={{ minWidth: 100 }}>
                <TextField
                  label="Temperature"
                  type="number"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  disabled={isLoading}
                  inputProps={{ min: 0.1, max: 1.0, step: 0.1 }}
                />
              </FormControl>

              {model === 'gpt2' && (
                <FormControl sx={{ minWidth: 100 }}>
                  <TextField
                    label="Top P"
                    type="number"
                    step="0.1"
                    value={topP}
                    onChange={(e) => setTopP(parseFloat(e.target.value))}
                    disabled={isLoading}
                    inputProps={{ min: 0.1, max: 1.0, step: 0.1 }}
                  />
                </FormControl>
              )}
            </Box>

            <Button
              type="submit"
              variant="contained"
              color="primary"
              disabled={isLoading || !prompt.trim() || !isBackendAvailable}
              fullWidth
              sx={{ py: 1.5, position: 'relative' }}
            >
              {isLoading ? (
                <>
                  <CircularProgress size={24} sx={{ position: 'absolute' }} />
                  <span style={{ opacity: 0 }}>Generating...</span>
                </>
              ) : 'Generate Response'}
            </Button>
          </form>
        </Paper>

        <Paper elevation={3} sx={{ mt: 4, p: 3, bgcolor: 'background.paper', borderRadius: 1 }}>
          <Typography variant="h6" gutterBottom>Response</Typography>
          <Box sx={{ 
            minHeight: '200px',
            p: 3, 
            bgcolor: 'background.default',
            borderRadius: 1,
            border: '1px solid',
            borderColor: 'divider',
            position: 'relative'
          }}>
            {isLoading ? (
              <Box sx={{ 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center', 
                justifyContent: 'center',
                height: '100%',
                color: 'text.secondary'
              }}>
                <CircularProgress sx={{ mb: 2 }} />
                <Typography variant="body1" align="center">
                  {response === 'Loading model and generating response. This may take a minute...' 
                    ? 'Loading model... (First time may take a few minutes)' 
                    : 'Generating response...'}
                </Typography>
                {response === 'Loading model and generating response. This may take a minute...' && (
                  <Typography variant="caption" color="textSecondary" align="center" sx={{ mt: 1, display: 'block' }}>
                    The model is being downloaded (about 2.7GB). This only happens once.
                  </Typography>
                )}
              </Box>
            ) : response ? (
              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                {response}
              </Typography>
            ) : (
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center', 
                height: '100%',
                color: 'text.secondary'
              }}>
                <Typography variant="body1">
                  Enter a prompt and click "Generate" to see the response.
                </Typography>
              </Box>
            )}
          </Box>
        </Paper>
        
        <Snackbar
          open={snackbarOpen}
          autoHideDuration={6000}
          onClose={handleCloseSnackbar}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert 
            onClose={handleCloseSnackbar} 
            severity={error ? 'error' : 'success'}
            sx={{ width: '100%' }}
          >
            {error || 'Request completed successfully'}
          </Alert>
        </Snackbar>
      </Container>
    </ThemeProvider>
  );
}

export default App;
