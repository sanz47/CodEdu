const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const crypto = require('crypto');
require('dotenv').config();

const app = express();

app.use(cors());
app.use(express.json());

// OAuth credentials (store these in environment variables)
const OAUTH_CONFIGS = {
  gmail: {
    clientId: process.env.GMAIL_CLIENT_ID,
    clientSecret: process.env.GMAIL_CLIENT_SECRET,
    tokenUrl: 'https://oauth2.googleapis.com/token',
    revokeUrl: 'https://oauth2.googleapis.com/revoke'
  },
  slack: {
    clientId: process.env.SLACK_CLIENT_ID,
    clientSecret: process.env.SLACK_CLIENT_SECRET,
    tokenUrl: 'https://slack.com/api/oauth.v2.access',
    revokeUrl: 'https://slack.com/api/auth.revoke'
  },
  notion: {
    clientId: process.env.NOTION_CLIENT_ID,
    clientSecret: process.env.NOTION_CLIENT_SECRET,
    tokenUrl: 'https://api.notion.com/v1/oauth/token'
  }
};

// Store tokens securely (in production, use a database with encryption)
const tokenStore = new Map();

// OAuth callback handlers for each service
app.post('/api/oauth/gmail/exchange', async (req, res) => {
  try {
    const { code, redirect_uri } = req.body;
    const config = OAUTH_CONFIGS.gmail;

    const tokenResponse = await fetch(config.tokenUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        client_id: config.clientId,
        client_secret: config.clientSecret,
        code,
        grant_type: 'authorization_code',
        redirect_uri
      })
    });

    const tokenData = await tokenResponse.json();
    
    if (tokenData.error) {
      return res.status(400).json({ error: tokenData.error_description });
    }

    // Store tokens securely (in production, associate with user ID)
    const userId = req.user?.id || 'demo-user'; // Get from authentication
    tokenStore.set(`gmail-${userId}`, {
      access_token: tokenData.access_token,
      refresh_token: tokenData.refresh_token,
      expires_at: Date.now() + (tokenData.expires_in * 1000)
    });

    res.json({ 
      access_token: tokenData.access_token,
      expires_in: tokenData.expires_in 
    });
  } catch (error) {
    console.error('Gmail OAuth exchange failed:', error);
    res.status(500).json({ error: 'OAuth exchange failed' });
  }
});

app.post('/api/oauth/slack/exchange', async (req, res) => {
  try {
    const { code, redirect_uri } = req.body;
    const config = OAUTH_CONFIGS.slack;

    const tokenResponse = await fetch(config.tokenUrl, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Basic ${Buffer.from(`${config.clientId}:${config.clientSecret}`).toString('base64')}`
      },
      body: new URLSearchParams({
        code,
        redirect_uri
      })
    });

    const tokenData = await tokenResponse.json();
    
    if (!tokenData.ok) {
      return res.status(400).json({ error: tokenData.error });
    }

    const userId = req.user?.id || 'demo-user';
    tokenStore.set(`slack-${userId}`, {
      access_token: tokenData.access_token,
      team_id: tokenData.team.id,
      team_name: tokenData.team.name
    });

    res.json({ 
      access_token: tokenData.access_token,
      team_name: tokenData.team.name 
    });
  } catch (error) {
    console.error('Slack OAuth exchange failed:', error);
    res.status(500).json({ error: 'OAuth exchange failed' });
  }
});

app.post('/api/oauth/notion/exchange', async (req, res) => {
  try {
    const { code, redirect_uri } = req.body;
    const config = OAUTH_CONFIGS.notion;

    const tokenResponse = await fetch(config.tokenUrl, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Basic ${Buffer.from(`${config.clientId}:${config.clientSecret}`).toString('base64')}`
      },
      body: JSON.stringify({
        grant_type: 'authorization_code',
        code,
        redirect_uri
      })
    });

    const tokenData = await tokenResponse.json();
    
    if (tokenData.error) {
      return res.status(400).json({ error: tokenData.error_description });
    }

    const userId = req.user?.id || 'demo-user';
    tokenStore.set(`notion-${userId}`, {
      access_token: tokenData.access_token,
      workspace_name: tokenData.workspace_name,
      workspace_id: tokenData.workspace_id
    });

    res.json({ 
      access_token: tokenData.access_token,
      workspace_name: tokenData.workspace_name 
    });
  } catch (error) {
    console.error('Notion OAuth exchange failed:', error);
    res.status(500).json({ error: 'OAuth exchange failed' });
  }
});

// API proxy endpoints to handle authenticated requests
app.get('/api/gmail/messages', async (req, res) => {
  try {
    const userId = req.user?.id || 'demo-user';
    const tokens = tokenStore.get(`gmail-${userId}`);
    
    if (!tokens) {
      return res.status(401).json({ error: 'Gmail not connected' });
    }

    // Check if token needs refresh
    if (Date.now() > tokens.expires_at && tokens.refresh_token) {
      await refreshGmailToken(userId, tokens.refresh_token);
    }

    const response = await fetch('https://gmail.googleapis.com/gmail/v1/users/me/messages?maxResults=10', {
      headers: { 'Authorization': `Bearer ${tokens.access_token}` }
    });

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Gmail API request failed:', error);
    res.status(500).json({ error: 'API request failed' });
  }
});

app.post('/api/gmail/send', async (req, res) => {
  try {
    const userId = req.user?.id || 'demo-user';
    const tokens = tokenStore.get(`gmail-${userId}`);
    const { to, subject, body } = req.body;
    
    if (!tokens) {
      return res.status(401).json({ error: 'Gmail not connected' });
    }

    const email = [
      `To: ${to}`,
      `Subject: ${subject}`,
      'Content-Type: text/plain; charset=utf-8',
      '',
      body
    ].join('\n');

    const response = await fetch('https://gmail.googleapis.com/gmail/v1/users/me/messages/send', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${tokens.access_token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        raw: Buffer.from(email).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
      })
    });

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Gmail send failed:', error);
    res.status(500).json({ error: 'Send failed' });
  }
});

app.post('/api/slack/message', async (req, res) => {
  try {
    const userId = req.user?.id || 'demo-user';
    const tokens = tokenStore.get(`slack-${userId}`);
    const { channel, text } = req.body;
    
    if (!tokens) {
      return res.status(401).json({ error: 'Slack not connected' });
    }

    const response = await fetch('https://slack.com/api/chat.postMessage', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${tokens.access_token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ channel, text })
    });

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Slack API request failed:', error);
    res.status(500).json({ error: 'API request failed' });
  }
});

app.post('/api/notion/page', async (req, res) => {
  try {
    const userId = req.user?.id || 'demo-user';
    const tokens = tokenStore.get(`notion-${userId}`);
    const { parent_id, title, content } = req.body;
    
    if (!tokens) {
      return res.status(401).json({ error: 'Notion not connected' });
    }

    const response = await fetch('https://api.notion.com/v1/pages', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${tokens.access_token}`,
        'Content-Type': 'application/json',
        'Notion-Version': '2022-06-28'
      },
      body: JSON.stringify({
        parent: { database_id: parent_id },
        properties: {
          Name: { title: [{ text: { content: title } }] }
        },
        children: [
          {
            object: 'block',
            type: 'paragraph',
            paragraph: {
              rich_text: [{ type: 'text', text: { content } }]
            }
          }
        ]
      })
    });

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Notion API request failed:', error);
    res.status(500).json({ error: 'API request failed' });
  }
});

// Token refresh function for Gmail
async function refreshGmailToken(userId, refreshToken) {
  try {
    const config = OAUTH_CONFIGS.gmail;
    const response = await fetch(config.tokenUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        client_id: config.clientId,
        client_secret: config.clientSecret,
        refresh_token: refreshToken,
        grant_type: 'refresh_token'
      })
    });

    const tokenData = await response.json();
    
    if (tokenData.access_token) {
      const existingTokens = tokenStore.get(`gmail-${userId}`);
      tokenStore.set(`gmail-${userId}`, {
        ...existingTokens,
        access_token: tokenData.access_token,
        expires_at: Date.now() + (tokenData.expires_in * 1000)
      });
    }
  } catch (error) {
    console.error('Token refresh failed:', error);
  }
}

// Webhook endpoints