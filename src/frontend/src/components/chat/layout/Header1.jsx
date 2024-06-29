import * as React from 'react';
import PropTypes from 'prop-types';
import AppBar from '@mui/material/AppBar';
import Grid from '@mui/material/Grid';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import Toolbar from '@mui/material/Toolbar';

import logo from '../../../assets/logo.svg'; // Import the SVG logo

function Header1(props) {
  const { onDrawerToggle } = props;

  return (
    <React.Fragment>
      <AppBar sx={{ backgroundColor: "#DA291C" }} position="sticky" elevation={0}>
        <Toolbar>
          <Grid container spacing={1} alignItems="center">
            <Grid sx={{ display: { sm: 'none', xs: 'block' } }} item>
              <IconButton
                color="inherit"
                aria-label="open drawer"
                onClick={onDrawerToggle}
                edge="start"
              >
                <MenuIcon />
              </IconButton>
            </Grid>
            <Grid item xs />
            <Grid item>
              <img src={logo} alt="Logo" style={{ height: 40 }} />
            </Grid>
          </Grid>
        </Toolbar>
      </AppBar>
    </React.Fragment>
  );
}

Header1.propTypes = {
  onDrawerToggle: PropTypes.func.isRequired,
};

export default Header1;