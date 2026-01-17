import http from 'http'
import fs from 'fs'
import { URL } from 'url'
import { config } from 'dotenv'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

// Load environment variables from .env file
const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
config({ path: join(__dirname, '..', '.env') })

const PORT = process.env.PORT || 3001
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || process.env.VITE_OPENROUTER_API_KEY

if (!OPENROUTER_API_KEY) {
  console.warn('Warning: OPENROUTER_API_KEY not set. Set OPENROUTER_API_KEY in your environment to call the OpenRouter API.')
}

function sendJson(res, status, obj) {
  const body = typeof obj === 'string' ? obj : JSON.stringify(obj)
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  })
  res.end(body)
}

const server = http.createServer(async (req, res) => {
  const { method, url } = req
  const parsed = new URL(url, `http://localhost:${PORT}`)
  console.log('[proxy] incoming', method, parsed.pathname)

  // Basic CORS preflight handling
  if (method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    })
    return res.end()
  }

  const pathNoSlash = parsed.pathname.replace(/\/+$/, '')

  // Serve built frontend from /dist when available (production)
  try {
    const distDir = join(__dirname, '..', 'dist')
    // Only attempt static serve for GET requests
    if (method === 'GET' && !parsed.pathname.startsWith('/api')) {
      // map root to index.html
      let relPath = parsed.pathname
      if (relPath === '/' || relPath === '') relPath = '/index.html'
      const filePath = join(distDir, relPath.replace(/^\//, ''))

      // Check if file exists
      if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
        const ext = filePath.split('.').pop() || ''
        const contentTypes = {
          html: 'text/html; charset=utf-8',
          js: 'application/javascript; charset=utf-8',
          css: 'text/css; charset=utf-8',
          json: 'application/json; charset=utf-8',
          png: 'image/png',
          jpg: 'image/jpeg',
          jpeg: 'image/jpeg',
          svg: 'image/svg+xml',
          ico: 'image/x-icon',
          map: 'application/octet-stream',
        }
        const body = fs.readFileSync(filePath)
        res.writeHead(200, {
          'Content-Type': contentTypes[ext] || 'application/octet-stream',
          'Access-Control-Allow-Origin': '*'
        })
        return res.end(body)
      }

      // SPA fallback - serve index.html for all non-API routes that don't match files
      const indexPath = join(distDir, 'index.html')
      if (fs.existsSync(indexPath)) {
        const body = fs.readFileSync(indexPath)
        res.writeHead(200, {
          'Content-Type': 'text/html; charset=utf-8',
          'Access-Control-Allow-Origin': '*'
        })
        return res.end(body)
      }
    }
  } catch (e) {
    console.warn('[static] serve error', e)
  }

  if (pathNoSlash === '/api/generate' && method === 'POST') {
    try {
      let body = ''
      for await (const chunk of req) body += chunk
      let json = {}
      try {
        json = body ? JSON.parse(body) : {}
      } catch (err) {
        console.warn('[proxy] failed to parse JSON body', err)
        json = {}
      }

      // Extract content from Gemini-format payload
      // Expecting: { contents: [{ parts: [{ text: "..." }] }] }
      const userMessage = json.contents?.[0]?.parts?.[0]?.text || "";

      if (!userMessage) {
        sendJson(res, 400, { error: 'Invalid payload format' });
        return;
      }

      // Forward request to OpenRouter API
      const forwardUrl = `https://openrouter.ai/api/v1/chat/completions`

      const openRouterPayload = {
        model: "xiaomi/mimo-v2-flash:free",
        messages: [
          { role: "user", content: userMessage }
        ]
      };

      console.log('Forwarding request to OpenRouter:', JSON.stringify(openRouterPayload, null, 2));

      const fetchRes = await fetch(forwardUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
          'HTTP-Referer': 'http://localhost:3000', // Best practice for OpenRouter
          'X-Title': 'Data Visualizer'
        },
        body: JSON.stringify(openRouterPayload)
      })

      if (!fetchRes.ok) {
        const errText = await fetchRes.text();
        console.error('OpenRouter error:', fetchRes.status, errText);
        throw new Error(`OpenRouter API error: ${fetchRes.status} ${errText}`);
      }

      const openRouterJson = await fetchRes.json();
      const assistantMessage = openRouterJson.choices?.[0]?.message?.content || "";

      // Transform back to Gemini format for frontend compatibility
      const geminiResponse = {
        candidates: [
          {
            content: {
              parts: [
                { text: assistantMessage }
              ]
            }
          }
        ]
      };

      sendJson(res, 200, geminiResponse);
    } catch (err) {
      console.error('Proxy error:', err);
      sendJson(res, 500, { error: 'Proxy error', details: String(err) });
    }
    return
  }

  // Basic health
  if (parsed.pathname === '/api/health') {
    return sendJson(res, 200, { ok: true })
  }

  // List available models (server-side) - using OpenRouter now
  if (parsed.pathname === '/api/list-models' && method === 'GET') {
    try {
      if (!OPENROUTER_API_KEY) return sendJson(res, 500, { error: 'OPENROUTER_API_KEY not configured on server' })

      const listUrl = `https://openrouter.ai/api/v1/models`
      const listRes = await fetch(listUrl, { method: 'GET', headers: { 'Authorization': `Bearer ${OPENROUTER_API_KEY}` } })
      const listBody = await listRes.text()

      res.writeHead(listRes.status, {
        'Content-Type': listRes.headers.get('content-type') || 'application/json',
        'Access-Control-Allow-Origin': '*'
      })
      return res.end(listBody)
    } catch (err) {
      console.error('List models error', err)
      return sendJson(res, 500, { error: 'Failed to list models', details: String(err) })
    }
  }

  // 404
  console.warn('[proxy] no route matched', method, parsed.pathname)
  sendJson(res, 404, { error: 'Not found', method, path: parsed.pathname })
})

server.listen(PORT, () => {
  console.log(`Proxy server listening on http://localhost:${PORT}`)
})
