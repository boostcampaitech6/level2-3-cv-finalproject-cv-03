import React, { useState } from "react";
import {
  StyleSheet,
  ImageBackground,
  Dimensions,
  StatusBar,
  KeyboardAvoidingView
} from "react-native";
import { Block, Checkbox, Text, theme } from "galio-framework";

import { Button, Input } from "../../components";
import { Images, argonTheme } from "../../constants";
import Icon from 'react-native-vector-icons/FontAwesome';
import { View } from 'react-native';

const { width, height } = Dimensions.get("screen");

const Register2 = (props) => {
  const { navigation } = props;
  const [isChecked2, setIsChecked2] = useState(false);
  const [isChecked3, setIsChecked3] = useState(false);
  const [dup, setDup] = useState(true);
  const [dup2, setDup2] = useState(false);

  const handleDup = async () => {
    setDup(false);
    setDup2(false);
    // try {
    //   const response = await fetch('http://34.64.33.83:3000/dup', {
    //     method: 'POST',
    //     headers: {
    //       'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify({ email }),
    //   });
  
    //   const data = await response.json();
    //   console.log(data);
    //   if (response.ok) {
    //     if (data.isDuplicate) {
    //       // 이메일이 이미 중복되는 경우
    //       setDup(true);
    //       setDup2(true);
    //     } else {
    //       // 이메일이 중복되지 않는 경우
    //       setDup(false);
    //       setDup2(false);
    //     }
    //   } else {
    //     console.log('Failed to check email duplication');
    //   }
    // } catch (error) {
    //   console.error('Network error:', error);
    // }
  };

  return (
    <Block flex middle>
      <StatusBar hidden />
      <ImageBackground
        source={Images.RegisterBackground}
        style={{ width, height, zIndex: 1 }}
      >
        <Block safe flex middle>
          <Block style={styles.registerContainer}>
            {/* <Block flex={0.25} middle style={styles.socialConnect}>
              <Text color="#8898AA" size={12}>
                Sign up with
              </Text>
              <Block row style={{ marginTop: theme.SIZES.BASE }}>
                <Button style={{ ...styles.socialButtons, marginRight: 30 }}>
                  <Block row>
                    <Icon
                      name="logo-github"
                      family="Ionicon"
                      size={14}
                      color={"black"}
                      style={{ marginTop: 2, marginRight: 5 }}
                    />
                    <Text style={styles.socialTextButtons}>GITHUB</Text>
                  </Block>
                </Button>
                <Button style={styles.socialButtons}>
                  <Block row>
                    <Icon
                      name="logo-google"
                      family="Ionicon"
                      size={14}
                      color={"black"}
                      style={{ marginTop: 2, marginRight: 5 }}
                    />
                    <Text style={styles.socialTextButtons}>GOOGLE</Text>
                  </Block>
                </Button>
              </Block>
            </Block> */}
            <Block flex>
              <Block flex={0.33} paddingLeft={30} style={{ justifyContent: 'flex-end' }}>
                <Text color="black" size={28} paddingBottom={20} style={styles.subTitle}>
                  회원가입
                </Text>
              </Block>
              <View paddingLeft={30} style={{ flexDirection: 'row'}}>
                <Text color={argonTheme.COLORS.MUTED} size={12} paddingBottom={10} style={styles.text}>
                  1. 약관 동의{" "} > {""}
                </Text>
                <Text color="black" size={10} paddingBottom={10} style={styles.text}>
                  2. 이메일 인증{" "}
                </Text>
                <Text color={argonTheme.COLORS.MUTED} size={10} paddingBottom={10} style={styles.text}>
                  > 3. 비밀번호 입력 > 4. URL 등록
                </Text>
              </View>
              <Block flex center>
                <KeyboardAvoidingView
                  style={{ flex: 1 }}
                  behavior="padding"
                  enabled
                >
                  <Text color="black" paddingBottom={20} style={styles.text} marginStart={10}>
                    이메일은 추후 변경이 불가능합니다.
                  </Text>
                  

                    <View marginTop={30} style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Block width={width * 0.55} >
                        <Input
                          borderless
                          placeholder="Email"
                          iconContent={
                            <Icon
                              size={16}
                              color={argonTheme.COLORS.ICON}
                              name="envelope-o"
                              style={styles.inputIcons}
                            />
                          }
                        />
                      </Block>
                      <Button 
                        onPress={handleDup}
                        color="button_color2"
                        style={{...styles.createButton, width: '25%', marginTop:6, marginRight:0}}
                        textStyle={{ fontSize: 13, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}
                      >
                        중복 확인
                      </Button>
                    </View>
                    {dup && (
                    <Text style={styles.text2} marginStart={5} color={argonTheme.COLORS.MUTED}>이메일 중복 확인을 해주세요.</Text>
                    )}
                    {dup2 && (
                    <Text style={styles.text2} marginStart={5} color={argonTheme.COLORS.ERROR}>중복된 이메일입니다.</Text>
                    )}
                  
                  
                  <Block middle marginTop={50}>
                    <Button 
                      onPress={() => navigation.navigate('Register3')}
                      color={!(dup || dup2) ? "primary" : "muted" } 
                      style={styles.createButton}
                      disabled={dup || dup2} // Button is disabled if either isChecked2 or isChecked3 is not checked
                      textStyle={{ fontSize: 13, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}
                    >
                      다음
                    </Button>
                  </Block>
                </KeyboardAvoidingView>
              </Block>
            </Block>
          </Block>
        </Block>
      </ImageBackground>
    </Block>
  );
}


const styles = StyleSheet.create({
  registerContainer: {
    width: width * 0.9,
    height: height * 0.875,
    backgroundColor: "#F4F5F7",
    borderRadius: 4,
    shadowColor: argonTheme.COLORS.BLACK,
    shadowOffset: {
      width: 0,
      height: 4
    },
    shadowRadius: 8,
    shadowOpacity: 0.1,
    elevation: 1,
    overflow: "hidden"
  },
  socialConnect: {
    backgroundColor: argonTheme.COLORS.WHITE,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: "#8898AA"
  },
  socialButtons: {
    width: 120,
    height: 40,
    backgroundColor: "#fff",
    shadowColor: argonTheme.COLORS.BLACK,
    shadowOffset: {
      width: 0,
      height: 4
    },
    shadowRadius: 8,
    shadowOpacity: 0.1,
    elevation: 1
  },
  socialTextButtons: {
    color: argonTheme.COLORS.PRIMARY,
    fontWeight: "800",
    fontSize: 14
  },
  inputIcons: {
    marginRight: 12
  },
  passwordCheck: {
    paddingLeft: 15,
    paddingTop: 13,
    paddingBottom: 30
  },
  createButton: {
    width: width * 0.5,
    marginTop: 25
  },
  subTitle: {
    fontFamily: 'SG',
    marginTop: 20
  },
  text: {
    fontFamily: 'NGB',
    fontSize: 10,
  },
  text2: {
    fontFamily: 'NGB',
    fontSize: 13,
  },
  text3: {
    fontFamily: 'NGB',
    fontSize: 12,
  },
});

export default Register2;
