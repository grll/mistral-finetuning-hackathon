import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';

import { Typography } from '@mui/material';
import FilterHdrIcon from '@mui/icons-material/FilterHdr';

const item = {
  py: '2px',
  px: 3,
  color: 'rgba(255, 255, 255, 0.7)',
  '&:hover, &:focus': {
    bgcolor: 'rgba(255, 255, 255, 0.08)',
  },
};

const itemCategory = {
  boxShadow: '0 -1px 0 rgb(255,255,255,0.1) inset',
  py: 1.5,
  px: 3,
};

export default function Navigator(props) {
  const { ...other } = props;
  return (
    <Drawer variant="permanent" sx={{ width: 300, '& .MuiDrawer-paper': { width: 300 } }} {...other}>
      <List disablePadding>
        <ListItem sx={{ ...item, ...itemCategory, fontSize: 22, color: '#fff' }}>
          <FilterHdrIcon sx={{ marginLeft: 3.5, marginRight: 2 }} /> Alplex
        </ListItem>
      </List>
      <Typography sx={{ p: 2, color: 'rgba(255, 255, 255, 0.7)', fontStyle: "bold" }}>How does it work?</Typography>
      <List disablePadding>
        <ListItem sx={item}>1. Type your legal situation.</ListItem>
        <ListItem sx={item}>2. Dona our legal assistant agent help clarifying your legal case.</ListItem>
        <ListItem sx={item}>3. Rachel our paralegal researcher agent find the most relevant law articles and provide a reply.</ListItem>
      </List>
    </Drawer>
  );
}