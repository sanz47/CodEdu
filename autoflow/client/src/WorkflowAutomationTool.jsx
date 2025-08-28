import React, { useState, useEffect } from 'react';
import { Plus, Mail, MessageSquare, FileText, Zap, Settings, Play, Pause, Trash2, Edit3, CheckCircle, Clock, AlertCircle, ExternalLink } from 'lucide-react';

// OAuth Configuration for each service (demo values - replace with your actual client IDs)
const OAUTH_CONFIGS = {
  gmail: {
    clientId: 'your-gmail-client-id.apps.googleusercontent.com',
    redirectUri: `${window.location.origin}/oauth/gmail/callback`,
    scope: 'https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.send',
    authUrl: 'https://accounts.google.com/o/oauth2/v2/auth'
  },
  slack: {
    clientId: 'your-slack-client-id',
    redirectUri: `${window.location.origin}/oauth/slack/callback`,
    scope: 'channels:read channels:write.public chat:write',
    authUrl: 'https://slack.com/oauth/v2/authorize'
  },
  notion: {
    clientId: 'your-notion-client-id',
    redirectUri: `${window.location.origin}/oauth/notion/callback`,
    authUrl: 'https://api.notion.com/v1/oauth/authorize'
  }
};

const WorkflowAutomationTool = () => {
  const [workflows, setWorkflows] = useState([
    {
      id: 1,
      name: "Email Digest & Auto-Reply",
      description: "Summarize unread emails and send auto-replies",
      apps: ["Gmail", "Slack"],
      triggers: "New email in inbox",
      actions: "Summarize + Send to Slack + Auto-reply",
      status: "active",
      lastRun: "2 hours ago",
      executions: 24
    },
    {
      id: 2,
      name: "Meeting Notes to Notion",
      description: "Extract action items from meeting transcripts",
      apps: ["Gmail", "Notion"],
      triggers: "Email with 'meeting notes'",
      actions: "Extract tasks + Create Notion pages",
      status: "active",
      lastRun: "1 day ago",
      executions: 8
    },
    {
      id: 3,
      name: "Weekly Report Generator",
      description: "Compile project updates into reports",
      apps: ["Slack", "Notion", "Gmail"],
      triggers: "Every Friday 5 PM",
      actions: "Gather updates + Generate report + Email team",
      status: "paused",
      lastRun: "1 week ago",
      executions: 12
    }
  ]);

  const [showCreateWorkflow, setShowCreateWorkflow] = useState(false);
  const [newWorkflow, setNewWorkflow] = useState({
    name: '',
    description: '',
    triggerApp: '',
    triggerEvent: '',
    actionApp: '',
    actionType: ''
  });

  const [activeTab, setActiveTab] = useState('workflows');
  const [analytics, setAnalytics] = useState({
    totalWorkflows: 3,
    activeWorkflows: 2,
    totalExecutions: 44,
    timeSaved: "12.5 hours"
  });

  // OAuth tokens stored in state (in production, use secure storage)
  const [oauthTokens, setOauthTokens] = useState({
    gmail: null, // Removed localStorage access for demo
    slack: null,
    notion: null
  });

  const apps = [
    { 
      name: 'Gmail', 
      icon: Mail, 
      color: 'bg-red-500', 
      connected: !!oauthTokens.gmail,
      apiKey: 'gmail',
      demoToken: 'demo-gmail-token'
    },
    { 
      name: 'Slack', 
      icon: MessageSquare, 
      color: 'bg-purple-500', 
      connected: !!oauthTokens.slack,
      apiKey: 'slack',
      demoToken: 'demo-slack-token'
    },
    { 
      name: 'Notion', 
      icon: FileText, 
      color: 'bg-gray-800', 
      connected: !!oauthTokens.notion,
      apiKey: 'notion',
      demoToken: 'demo-notion-token'
    },
    { 
      name: 'Trello', 
      icon: FileText, 
      color: 'bg-blue-500', 
      connected: false,
      apiKey: 'trello'
    }
  ];

  // OAuth Functions - For demo purposes, we'll simulate the connection
  const initiateOAuth = (service) => {
    // In a real implementation, this would open OAuth flow
    // For demo, we'll simulate successful connection
    const demoToken = `demo-${service}-token-${Date.now()}`;
    
    // Simulate OAuth success after 1 second
    setTimeout(() => {
      setOauthTokens(prev => ({ ...prev, [service]: demoToken }));
      alert(`${service.charAt(0).toUpperCase() + service.slice(1)} connected successfully!\n\nIn a real implementation, this would:\n1. Open OAuth popup\n2. User grants permissions\n3. Exchange code for access token\n4. Store token securely`);
    }, 1000);
  };

  const handleOAuthCallback = async (service, code, state) => {
    // This would be handled by your backend in production
    console.log(`OAuth callback for ${service} with code: ${code}`);
  };

  // API Functions for each service - Demo implementations
  const gmailAPI = {
    async getMessages() {
      if (!oauthTokens.gmail) throw new Error('Gmail not connected');
      // Demo response - in real implementation, this would call Gmail API
      return {
        messages: [
          { id: '1', snippet: 'Meeting tomorrow at 3pm' },
          { id: '2', snippet: 'Project update needed' },
          { id: '3', snippet: 'Invoice for services' }
        ]
      };
    },
    
    async sendMessage(to, subject, body) {
      if (!oauthTokens.gmail) throw new Error('Gmail not connected');
      // Demo response
      return { id: 'demo-message-id', success: true };
    }
  };

  const slackAPI = {
    async postMessage(channel, text) {
      if (!oauthTokens.slack) throw new Error('Slack not connected');
      // Demo response
      return { ok: true, message: { text, channel } };
    },

    async getChannels() {
      if (!oauthTokens.slack) throw new Error('Slack not connected');
      // Demo response
      return {
        channels: [
          { id: 'C123', name: 'general' },
          { id: 'C456', name: 'dev-team' }
        ]
      };
    }
  };

  const notionAPI = {
    async createPage(parent_id, title, content) {
      if (!oauthTokens.notion) throw new Error('Notion not connected');
      // Demo response
      return { id: 'demo-page-id', title, content };
    },

    async getDatabases() {
      if (!oauthTokens.notion) throw new Error('Notion not connected');
      // Demo response
      return {
        results: [
          { id: 'db1', title: [{ text: { content: 'Tasks' } }] },
          { id: 'db2', title: [{ text: { content: 'Projects' } }] }
        ]
      };
    }
  };

  // Example workflow execution
  const executeWorkflow = async (workflowId) => {
    const workflow = workflows.find(w => w.id === workflowId);
    if (!workflow) return;

    try {
      switch (workflow.name) {
        case "Email Digest & Auto-Reply":
          const messages = await gmailAPI.getMessages();
          const summary = `Found ${messages.messages?.length || 0} new emails`;
          await slackAPI.postMessage('#general', `Email Digest: ${summary}`);
          break;
          
        case "Meeting Notes to Notion":
          const meetings = await gmailAPI.getMessages();
          // Process meeting notes and create Notion pages
          await notionAPI.createPage('database-id', 'Meeting Notes', 'Action items from today');
          break;
          
        default:
          console.log(`Executing workflow: ${workflow.name}`);
      }
      
      // Update workflow execution count
      setWorkflows(prev => prev.map(w => 
        w.id === workflowId 
          ? { ...w, executions: w.executions + 1, lastRun: 'Just now' }
          : w
      ));
    } catch (error) {
      console.error('Workflow execution failed:', error);
    }
  };

  // Listen for OAuth callbacks
  useEffect(() => {
    const handleMessage = (event) => {
      if (event.data.type === 'OAUTH_CALLBACK') {
        handleOAuthCallback(event.data.service, event.data.code, event.data.state);
      }
    };
    
    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  const triggerEvents = {
    Gmail: ['New email received', 'Email with keyword', 'Email from specific sender'],
    Slack: ['New message in channel', 'Mention received', 'File uploaded'],
    Notion: ['New page created', 'Database updated', 'Property changed']
  };

  const actionTypes = {
    Gmail: ['Send email', 'Create draft', 'Add label', 'Forward email'],
    Slack: ['Send message', 'Create channel', 'Update status', 'Post to channel'],
    Notion: ['Create page', 'Update database', 'Add comment', 'Create task']
  };

  const toggleWorkflowStatus = (id) => {
    setWorkflows(prev => prev.map(workflow => 
      workflow.id === id 
        ? { ...workflow, status: workflow.status === 'active' ? 'paused' : 'active' }
        : workflow
    ));
  };

  const deleteWorkflow = (id) => {
    setWorkflows(prev => prev.filter(workflow => workflow.id !== id));
  };

  const createWorkflow = () => {
    if (newWorkflow.name && newWorkflow.triggerApp && newWorkflow.actionApp) {
      const workflow = {
        id: Date.now(),
        name: newWorkflow.name,
        description: newWorkflow.description,
        apps: [newWorkflow.triggerApp, newWorkflow.actionApp],
        triggers: newWorkflow.triggerEvent,
        actions: newWorkflow.actionType,
        status: 'active',
        lastRun: 'Never',
        executions: 0
      };
      setWorkflows(prev => [...prev, workflow]);
      setNewWorkflow({ name: '', description: '', triggerApp: '', triggerEvent: '', actionApp: '', actionType: '' });
      setShowCreateWorkflow(false);
    }
  };

  const StatusIcon = ({ status }) => {
    if (status === 'active') return <CheckCircle className="w-4 h-4 text-green-500" />;
    if (status === 'paused') return <Pause className="w-4 h-4 text-yellow-500" />;
    return <AlertCircle className="w-4 h-4 text-red-500" />;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-lg border-b">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">AutoFlow AI</h1>
                <p className="text-sm text-gray-600">Intelligent Workflow Automation</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm text-gray-600">Time Saved This Month</p>
                <p className="text-lg font-bold text-green-600">{analytics.timeSaved}</p>
              </div>
              <button className="p-2 text-gray-600 hover:text-gray-900 transition-colors">
                <Settings className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Analytics Dashboard */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-xl shadow-lg border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Workflows</p>
                <p className="text-2xl font-bold text-gray-900">{analytics.totalWorkflows}</p>
              </div>
              <div className="p-3 bg-blue-100 rounded-lg">
                <Zap className="w-6 h-6 text-blue-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-lg border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Active Workflows</p>
                <p className="text-2xl font-bold text-green-600">{analytics.activeWorkflows}</p>
              </div>
              <div className="p-3 bg-green-100 rounded-lg">
                <CheckCircle className="w-6 h-6 text-green-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-lg border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Executions</p>
                <p className="text-2xl font-bold text-purple-600">{analytics.totalExecutions}</p>
              </div>
              <div className="p-3 bg-purple-100 rounded-lg">
                <Play className="w-6 h-6 text-purple-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-lg border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Success Rate</p>
                <p className="text-2xl font-bold text-indigo-600">98.2%</p>
              </div>
              <div className="p-3 bg-indigo-100 rounded-lg">
                <Clock className="w-6 h-6 text-indigo-600" />
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 bg-white p-1 rounded-lg shadow-lg border mb-8">
          <button
            onClick={() => setActiveTab('workflows')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'workflows'
                ? 'bg-blue-500 text-white'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Workflows
          </button>
          <button
            onClick={() => setActiveTab('apps')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'apps'
                ? 'bg-blue-500 text-white'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Connected Apps
          </button>
        </div>

        {/* Workflows Tab */}
        {activeTab === 'workflows' && (
          <div>
            {/* Create Workflow Button */}
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold text-gray-900">Your Workflows</h2>
              <button
                onClick={() => setShowCreateWorkflow(true)}
                className="flex items-center space-x-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white px-4 py-2 rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all shadow-lg"
              >
                <Plus className="w-4 h-4" />
                <span>Create Workflow</span>
              </button>
            </div>

            {/* Workflows List */}
            <div className="space-y-4">
              {workflows.map((workflow) => (
                <div key={workflow.id} className="bg-white p-6 rounded-xl shadow-lg border hover:shadow-xl transition-shadow">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <StatusIcon status={workflow.status} />
                      <h3 className="text-lg font-semibold text-gray-900">{workflow.name}</h3>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        workflow.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {workflow.status}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => toggleWorkflowStatus(workflow.id)}
                        className={`p-2 rounded-lg transition-colors ${
                          workflow.status === 'active'
                            ? 'text-yellow-600 hover:bg-yellow-100'
                            : 'text-green-600 hover:bg-green-100'
                        }`}
                      >
                        {workflow.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      </button>
                      <button className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors">
                        <Edit3 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => deleteWorkflow(workflow.id)}
                        className="p-2 text-red-600 hover:bg-red-100 rounded-lg transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  
                  <p className="text-gray-600 mb-4">{workflow.description}</p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <div>
                      <p className="text-sm font-medium text-gray-700">Connected Apps</p>
                      <div className="flex space-x-1 mt-1">
                        {workflow.apps.map((app, index) => (
                          <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded">
                            {app}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-700">Trigger</p>
                      <p className="text-sm text-gray-600 mt-1">{workflow.triggers}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-700">Actions</p>
                      <p className="text-sm text-gray-600 mt-1">{workflow.actions}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm text-gray-600">
                    <span>Last run: {workflow.lastRun}</span>
                    <span>{workflow.executions} executions</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Connected Apps Tab */}
        {activeTab === 'apps' && (
          <div>
            <h2 className="text-xl font-bold text-gray-900 mb-6">Connected Applications</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {apps.map((app) => {
                const IconComponent = app.icon;
                return (
                  <div key={app.name} className="bg-white p-6 rounded-xl shadow-lg border">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <div className={`p-3 ${app.color} rounded-lg`}>
                          <IconComponent className="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-gray-900">{app.name}</h3>
                          <p className={`text-sm ${app.connected ? 'text-green-600' : 'text-gray-500'}`}>
                            {app.connected ? 'Connected' : 'Not connected'}
                          </p>
                        </div>
                      </div>
                      <div className={`w-3 h-3 rounded-full ${app.connected ? 'bg-green-500' : 'bg-gray-300'}`}></div>
                    </div>
                    
                    {app.connected ? (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Workflows using this app:</span>
                          <span className="font-medium">
                            {workflows.filter(w => w.apps.includes(app.name)).length}
                          </span>
                        </div>
                        <div className="space-y-2">
                          <button className="w-full py-2 px-4 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors">
                            Manage Connection
                          </button>
                          <button 
                            onClick={() => executeWorkflow(workflows[0]?.id)}
                            className="w-full py-2 px-4 bg-green-500 text-white rounded-lg text-sm font-medium hover:bg-green-600 transition-colors flex items-center justify-center space-x-1"
                          >
                            <Play className="w-3 h-3" />
                            <span>Test API</span>
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <button
                          onClick={() => initiateOAuth(app.apiKey)}
                          className="w-full py-2 px-4 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 transition-colors flex items-center justify-center space-x-1"
                        >
                          <ExternalLink className="w-3 h-3" />
                          <span>Connect {app.name}</span>
                        </button>
                        <div className="text-xs text-gray-500 text-center">
                          Secure OAuth 2.0 connection
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Create Workflow Modal */}
        {showCreateWorkflow && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
              <div className="p-6 border-b">
                <h2 className="text-xl font-bold text-gray-900">Create New Workflow</h2>
                <p className="text-gray-600 mt-1">Automate your repetitive tasks with AI-powered integrations</p>
              </div>
              
              <div className="p-6 space-y-6">
                {/* API Setup Guide */}
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                  <h3 className="font-semibold text-blue-900 mb-2">🔧 Real API Implementation Guide</h3>
                  <p className="text-blue-800 text-sm mb-3">This is a demo showing the UI. For production, you'll need:</p>
                  <div className="space-y-2 text-sm">
                    <div>
                      <strong>Gmail API:</strong> Google Cloud Console → APIs & Services → Create OAuth 2.0 credentials
                      <br />
                      <span className="text-xs text-blue-600">Scopes: gmail.readonly, gmail.send</span>
                    </div>
                    <div>
                      <strong>Slack API:</strong> api.slack.com → Create New App → OAuth & Permissions
                      <br />
                      <span className="text-xs text-blue-600">Scopes: channels:read, chat:write</span>
                    </div>
                    <div>
                      <strong>Notion API:</strong> notion.so/my-integrations → New Integration
                      <br />
                      <span className="text-xs text-blue-600">Internal integration with page permissions</span>
                    </div>
                  </div>
                  <div className="mt-3 p-2 bg-yellow-100 rounded text-xs">
                    <strong>Demo Mode:</strong> Connections are simulated. Click "Connect" to see the flow!
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Workflow Name
                  </label>
                  <input
                    type="text"
                    value={newWorkflow.name}
                    onChange={(e) => setNewWorkflow({...newWorkflow, name: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="e.g., Email Summarizer"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Description
                  </label>
                  <textarea
                    value={newWorkflow.description}
                    onChange={(e) => setNewWorkflow({...newWorkflow, description: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    rows="3"
                    placeholder="What does this workflow do?"
                  />
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Trigger App
                    </label>
                    <select
                      value={newWorkflow.triggerApp}
                      onChange={(e) => setNewWorkflow({...newWorkflow, triggerApp: e.target.value, triggerEvent: ''})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="">Select trigger app</option>
                      {apps.filter(app => app.connected).map(app => (
                        <option key={app.name} value={app.name}>{app.name}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Trigger Event
                    </label>
                    <select
                      value={newWorkflow.triggerEvent}
                      onChange={(e) => setNewWorkflow({...newWorkflow, triggerEvent: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      disabled={!newWorkflow.triggerApp}
                    >
                      <option value="">Select trigger event</option>
                      {newWorkflow.triggerApp && triggerEvents[newWorkflow.triggerApp]?.map(event => (
                        <option key={event} value={event}>{event}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Action App
                    </label>
                    <select
                      value={newWorkflow.actionApp}
                      onChange={(e) => setNewWorkflow({...newWorkflow, actionApp: e.target.value, actionType: ''})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="">Select action app</option>
                      {apps.filter(app => app.connected).map(app => (
                        <option key={app.name} value={app.name}>{app.name}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Action Type
                    </label>
                    <select
                      value={newWorkflow.actionType}
                      onChange={(e) => setNewWorkflow({...newWorkflow, actionType: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      disabled={!newWorkflow.actionApp}
                    >
                      <option value="">Select action type</option>
                      {newWorkflow.actionApp && actionTypes[newWorkflow.actionApp]?.map(action => (
                        <option key={action} value={action}>{action}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              
              <div className="p-6 border-t flex space-x-4">
                <button
                  onClick={() => setShowCreateWorkflow(false)}
                  className="flex-1 py-2 px-4 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={createWorkflow}
                  className="flex-1 py-2 px-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all"
                >
                  Create Workflow
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default WorkflowAutomationTool;