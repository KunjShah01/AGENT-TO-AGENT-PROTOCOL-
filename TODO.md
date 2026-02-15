# A2A Protocol Extension Plan

## Overview

Extend the existing A2A protocol with new methods and features to create a custom A2A style protocol.

## Current Implementation

- `tasks/send` - Send a task to an agent
- `tasks/status` - Get task status
- `tasks/cancel` - Cancel a task

## Planned Extensions

### 1. New A2A Protocol Methods

- [x] `tasks/submit` - Submit a task (enhanced version of tasks/send)
- [x] `tasks/get` - Get detailed task information
- [x] `tasks/subscribe` - Subscribe to task events (SSE)
- [x] `messages/history` - Get message history between agents
- [x] `messages/send` - Send a message (not just tasks)
- [x] `agents/discover` - Discover agents by capability
- [x] `agents/list` - List all available agents
- [x] `agents/health` - Get agent health status
- [x] `agents/capabilities` - Get agent capabilities

### 2. Streaming Support

- [ ] Add Server-Sent Events (SSE) support for real-time task updates
- [ ] Add streaming response for long-running tasks

### 3. Enhanced Features

- [ ] Webhook/callback support for async notifications
- [ ] Task batching support
- [ ] Message correlation/threading

## Files to Modify

- `src/protocols/a2a_handler.py` - Add new protocol methods
- `src/api/endpoints/messages.py` - Add new API endpoints
- `src/core/message.py` - Add new message types if needed

## Follow-up Steps

1. Implement new protocol methods in A2AHandler
2. Add new API endpoints
3. Test the new protocol methods
