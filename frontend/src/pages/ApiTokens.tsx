import { useEffect, useRef, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material'
import { ContentCopy } from '@mui/icons-material'
import { api, ApiToken } from '../services/api'

const TOOLS = '["list_projects", "search_documents", "get_document", "get_wiki_page", "related_entities"]'

function hermesConfig(mcpUrl: string, token: string) {
  return `mcp_servers:
  docharvester:
    url: "${mcpUrl}"
    headers:
      Authorization: "Bearer ${token}"
    tools:
      include: ${TOOLS}`
}

function errorText(err: any) {
  return err?.response?.data?.detail || err?.message || 'Something went wrong'
}

function CopyBlock({ label, text }: { label: string; text: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
  }
  return (
    <Box>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
        <Typography variant="subtitle2">{label}</Typography>
        <Button startIcon={<ContentCopy />} onClick={copy} sx={{ minHeight: 48 }}>
          {copied ? 'Copied' : 'Copy'}
        </Button>
      </Stack>
      <Box
        component="pre"
        sx={{ m: 0, p: 2, bgcolor: 'grey.100', borderRadius: 1, overflowX: 'auto', fontSize: 14 }}
      >
        {text}
      </Box>
    </Box>
  )
}

export default function ApiTokens() {
  const [tokens, setTokens] = useState<ApiToken[]>([])
  const [name, setName] = useState('hermes')
  const [created, setCreated] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [toRevoke, setToRevoke] = useState<ApiToken | null>(null)
  const cancelRef = useRef<HTMLButtonElement>(null)
  const [mcp, setMcp] = useState({ mcp_url: 'http://localhost:8000/mcp', shared: false })

  const load = async () => {
    try {
      setTokens(await api.getApiTokens())
    } catch (err) {
      setError(`Couldn't load tokens: ${errorText(err)}`)
    }
  }

  useEffect(() => {
    load()
    api.getMcpInfo().then(setMcp).catch(() => {})
  }, [])

  const create = async () => {
    setError(null)
    try {
      const result = await api.createApiToken(name.trim() || 'hermes')
      setCreated(result.token)
      await load()
    } catch (err) {
      setError(`Couldn't create the token: ${errorText(err)}`)
    }
  }

  const revoke = async () => {
    if (!toRevoke) return
    setError(null)
    try {
      await api.revokeApiToken(toRevoke.id)
      await load()
    } catch (err) {
      setError(`Couldn't revoke the token: ${errorText(err)}`)
    } finally {
      setToRevoke(null)
    }
  }

  // The API sends UTC times without a zone marker. Mark them as UTC so the browser shows local time.
  const when = (value?: string | null) => {
    if (!value) return 'Never'
    const hasZone = /([zZ]|[+-]\d\d:?\d\d)$/.test(value)
    return new Date(hasZone ? value : `${value}Z`).toLocaleString()
  }

  return (
    <Stack spacing={3} sx={{ maxWidth: 820 }}>
      <Box>
        <Typography variant="h5" gutterBottom>API tokens</Typography>
        <Typography color="text.secondary">
          Let Hermes Agent (or another MCP client) read your projects. It sees only what you can see.
        </Typography>
      </Box>

      {!mcp.shared && (
        <Alert severity="info">
          Hermes on another computer? On this computer, run <b>scripts/connect-tailscale</b> once.
        </Alert>
      )}

      {error && <Alert severity="error" onClose={() => setError(null)}>{error}</Alert>}

      <Paper sx={{ p: 3 }}>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems={{ sm: 'center' }}>
          <TextField
            label="Token name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            inputProps={{ maxLength: 100 }}
            sx={{ flexGrow: 1 }}
          />
          <Button variant="contained" size="large" onClick={create} sx={{ minHeight: 48 }}>
            Create token
          </Button>
        </Stack>
      </Paper>

      {created && (
        <Paper sx={{ p: 3 }}>
          <Stack spacing={2}>
            <Alert severity="warning">Copy it now. You won't see this token again.</Alert>
            <CopyBlock label="Token" text={created} />
            <CopyBlock label="Paste into ~/.hermes/config.yaml, then restart Hermes" text={hermesConfig(mcp.mcp_url, created)} />
            <Button onClick={() => setCreated(null)} sx={{ alignSelf: 'flex-start', minHeight: 48 }}>
              Done, I saved it
            </Button>
          </Stack>
        </Paper>
      )}

      <Paper>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Last used</TableCell>
              <TableCell align="right">Status</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {tokens.length === 0 && (
              <TableRow>
                <TableCell colSpan={4}>No tokens yet. Create one above.</TableCell>
              </TableRow>
            )}
            {tokens.map((t) => (
              <TableRow key={t.id}>
                <TableCell>{t.name}</TableCell>
                <TableCell>{when(t.created_at)}</TableCell>
                <TableCell>{when(t.last_used_at)}</TableCell>
                <TableCell align="right">
                  {t.revoked_at ? (
                    <Typography color="text.secondary">Revoked</Typography>
                  ) : (
                    <Button color="error" onClick={() => setToRevoke(t)} sx={{ minHeight: 48 }}>
                      Revoke
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>

      <Dialog
        open={Boolean(toRevoke)}
        onClose={() => setToRevoke(null)}
        // Cancel gets focus once the dialog is open, so Enter never revokes by accident
        TransitionProps={{ onEntered: () => cancelRef.current?.focus() }}
      >
        <DialogTitle>Revoke "{toRevoke?.name}"?</DialogTitle>
        <DialogContent>
          <Typography>Anything using this token stops working right away.</Typography>
        </DialogContent>
        <DialogActions>
          <Button ref={cancelRef} onClick={() => setToRevoke(null)}>Cancel</Button>
          <Button color="error" onClick={revoke}>Revoke</Button>
        </DialogActions>
      </Dialog>
    </Stack>
  )
}
