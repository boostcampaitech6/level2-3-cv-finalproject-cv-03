import React from "react";
import {
  ImageBackground,
  Image,
  StyleSheet,
  StatusBar,
  Dimensions
} from "react-native";
import * as Font from 'expo-font';
import { AppLoading } from 'expo';
import { Block, Button, Text, theme } from "galio-framework";

const { height, width } = Dimensions.get("screen");

import argonTheme from "../../constants/Theme";
import Images from "../../constants/Images";


class Onboarding extends React.Component {
  render() {
    const { navigation } = this.props;

    return (
      <Block flex style={styles.container}>
        <StatusBar hidden />
        <Block flex center>
        <ImageBackground
            source={Images.Onboarding}
            style={{ height, width, zIndex: 1 }}
          />
        </Block>
        <Block center>
          <Image source={Images.LogoOnboarding} style={styles.logo} resizeMode="contain"/>
        </Block>
        <Block flex space="between" style={styles.padded}>
            <Block flex space="around" style={{ zIndex: 2 }}>
              <Block style={styles.title}>
                <Block>
                  <Text color="white" size={60} style={styles.title}>
                    와치덕
                  </Text>
                </Block>
                <Block>
                  <Text color="white" size={60} style={styles.title}>
                    
                  </Text>
                </Block>
                <Block style={styles.subTitle}>
                  <Text color="white" size={16} style={styles.subTitle}>
                    와치덕과 함께라면 도난 걱정 NO!
                  </Text>
                  <Text color="white" size={16} style={styles.subTitle}>
                    CCTV 실시간 모니터링으로 안심하세요.
                  </Text>
                </Block>
              </Block>
              <Block center>
                <Button
                  style={styles.button}
                  color={argonTheme.COLORS.SECONDARY}
                  onPress={() => navigation.navigate("Login")}
                  textStyle={{ color: argonTheme.COLORS.BLACK, fontFamily: 'NGB',}}
                >
                  시작하기
                </Button>
              </Block>
          </Block>
        </Block>
      </Block>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.COLORS.BLACK
  },
  padded: {
    paddingHorizontal: theme.SIZES.BASE * 2,
    position: "relative",
    bottom: theme.SIZES.BASE,
    zIndex: 2,
  },
  button: {
    width: width - theme.SIZES.BASE * 4,
    height: theme.SIZES.BASE * 3,
    shadowRadius: 0,
    shadowOpacity: 0
  },
  logo: {
    width: 150,
    height: 150,
    zIndex: 2,
    position: 'relative',
    marginTop: '-50%'
  },
  title: {
    fontFamily: 'SG',
    marginTop:'0%'
  },
  subTitle: {
    fontFamily: 'NGB',
    marginTop: 10
  }
});

export default Onboarding;
