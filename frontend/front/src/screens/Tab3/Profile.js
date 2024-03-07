import React, { useState, useEffect, useContext } from "react";
import {
  StyleSheet,
  Dimensions,
  ScrollView,
  Image,
  ImageBackground,
  Platform
} from "react-native";
import { Block, Text, theme } from "galio-framework";
import { useFocusEffect } from '@react-navigation/native';

import { Button, Input } from "../../components";
import { Images, argonTheme } from "../../constants";
import { HeaderHeight } from "../../constants/utils";
import { UserContext } from '../../UserContext';
import { Icon, Overlay } from 'react-native-elements';
import { View } from 'react-native';



const { width, height } = Dimensions.get("screen");

const thumbMeasure = (width - 48 - 32) / 3;

const Profile = (props) => {
  const { navigation } = props;
  const { user } = useContext(UserContext);
  // console.log(user)
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [store_name, setStore_name] = useState("");
  const [npassword, setNpassword] = useState(password);
  const [nstore_name, setNstore_name] = useState(store_name);
  const [overlayVisible, setOverlayVisible] = useState(false);
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [password2Visible, setPassword2Visible] = useState(false);
  const [inpassword, setInpassword] = useState("");

  useEffect(() => {
    setNstore_name(store_name);
  }, [store_name]);

  useFocusEffect(
    React.useCallback(() => {
      const fetchData = async () => {
        try {
          const response = await fetch(`http://10.28.224.142:30016/api/v0/settings/profile_lookup?member_id=${user}`, {
            method: "GET",
            headers: { 'accept': 'application/json' },
          });
          const data = await response.json();
          // console.log(data);
          if (data.isSuccess) {
            setEmail(data.result.email);
            setPassword(data.result.password);
            setStore_name(data.result.store_name);
          }
          else {
            console.error('No information:', error);
          }
        } catch (error) {
          console.error('Network error:', error);
        }
      }
  
      fetchData();
    }, [user])
  );

  return (
    <Block flex style={styles.profile}>
      <Block flex>
        <ImageBackground
          source={Images.ProfileBackground}
          style={styles.profileContainer}
          imageStyle={styles.profileBackground}
        >
          <ScrollView
            showsVerticalScrollIndicator={false}
            style={{ width, marginTop: '30%' }}
          >
            <Block flex style={styles.profileCard}>
              <Block middle style={styles.avatarContainer}>
                <Image
                  source={{ uri: Images.ProfilePicture }}
                  style={styles.avatar}
                />
              </Block>
              <Block style={styles.info}>
              </Block>
              <Block flex>
                <Block middle style={styles.nameInfo}>
                  <Text bold size={28} color="#32325D" style={styles.textName}>
                    홍길동
                  </Text>
                  <Text size={16} color="#32325D" style={{ ...styles.text, marginTop: 10 }} >
                    사장님
                  </Text>
                </Block>
                <Block middle style={{ marginTop: 30, marginBottom: 16 }}>
                  <Block style={styles.divider} />
                </Block>
                <Block middle style={{marginTop: 20}} row space="between">
                  <Text bold size={16} color="#525F7F" style={styles.text}>
                    이메일
                  </Text>
                  <Text bold size={16} color="#525F7F" style={styles.text2}>
                    {email}
                  </Text>
                </Block>
                <Block middle style={{marginTop: 20}} row space="between">
                  <Text bold size={16} color="#525F7F" style={styles.text}>
                    PW
                  </Text>
                  <View style={{ flexDirection: 'row', alignItems: 'center', marginRight: 8 }}>
                    <Text bold size={16} color="#525F7F" style={styles.text2}>
                      **********
                    </Text>
                    <Icon
                      name='pencil'
                      size={20}
                      type='font-awesome'
                      onPress={() => setPasswordVisible(true)}
                    />
                  </View>
                </Block>
                <Block middle style={{marginTop: 20}} row space="between">
                  <Text bold size={16} color="#525F7F" style={styles.text}>
                    매장명
                  </Text>
                  <View style={{ flexDirection: 'row', alignItems: 'center', marginRight: 8 }}>
                    <Text bold size={16} color="#525F7F" style={styles.text2}>
                      {store_name}
                    </Text>
                    <Icon
                      name='pencil'
                      size={20}
                      type='font-awesome'
                      onPress={() => setOverlayVisible(true)}
                    />
                  </View>
                </Block>
                <Overlay isVisible={overlayVisible} onBackdropPress={() => setOverlayVisible(false)}>
                  <View style={{ alignItems: 'center', justifyContent: 'center', padding: 40 }}>
                    <Text style={styles.poptitle}>매장명 수정</Text>
                    <Input value={nstore_name} defaultValue={store_name} onChangeText={setNstore_name} placeholder={"새 매장명"}/>
                    <Button style={{marginTop:20,}} color="primary" onPress={() => {
                      setNpassword(nstore_name);
                      setOverlayVisible(false);
                    }} >
                      <Text style={{ fontSize: 14, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}>
                        저장하기
                      </Text>
                    </Button>
                  </View>
                </Overlay>
                <Overlay isVisible={passwordVisible} onBackdropPress={() => setPasswordVisible(false)}>
                  <View style={{ alignItems: 'center', justifyContent: 'center', padding: 40 }}>
                    <Text style={styles.poptitle}>비밀번호 확인</Text>
                    <Input value={inpassword} onChangeText={setInpassword} placeholder={"기존 비밀번호"}/>
                    <Button style={{marginTop:20,}} color="primary" onPress={() => {
                      setInpassword(inpassword);
                      setPasswordVisible(false);
                      setPassword2Visible(true);
                    }} >
                      <Text style={{ fontSize: 14, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}>
                        다음
                      </Text>
                    </Button>
                  </View>
                </Overlay>
                <Overlay isVisible={password2Visible} onBackdropPress={() => setPassword2Visible(false)}>
                  <View style={{ alignItems: 'center', justifyContent: 'center', padding: 40 }}>
                    <Text style={styles.poptitle}>새 비밀번호</Text>
                    <Input value={npassword} onChangeText={setNpassword} placeholder={"새 비밀번호"}/>
                    <Button style={{marginTop:20,}} color="primary" onPress={() => {
                      setNpassword(npassword);
                      setPassword2Visible(false);
                    }} >
                      <Text style={{ fontSize: 14, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}>
                        저장하기
                      </Text>
                    </Button>
                  </View>
                </Overlay>
                {/* <Block middle marginTop={50}>
                  <Button 
                    onPress={() => navigation.navigate('ProfileEdit', { email: email, password: password, store_name: store_name })}
                    color={ "primary" } 
                    style={styles.createButton}
                    textStyle={{ fontSize: 13, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}
                  >
                    수정하기
                  </Button>
                </Block> */}
              </Block>
            </Block>
          </ScrollView>
        </ImageBackground>
      </Block>
    </Block>
  );
}


const styles = StyleSheet.create({
  profile: {
    marginTop: Platform.OS === "android" ? -HeaderHeight : 0,
    // marginBottom: -HeaderHeight * 2,
    flex: 1
  },
  profileContainer: {
    width: width,
    height: height,
    padding: 0,
    zIndex: 1
  },
  profileBackground: {
    width: width,
    height: height/2
  },
  profileCard: {
    // position: "relative",
    padding: theme.SIZES.BASE,
    marginHorizontal: theme.SIZES.BASE,
    marginTop: 65,
    borderTopLeftRadius: 6,
    borderTopRightRadius: 6,
    borderRadius: 6,
    backgroundColor: theme.COLORS.WHITE,
    shadowColor: "black",
    shadowOffset: { width: 0, height: 0 },
    shadowRadius: 8,
    shadowOpacity: 0.2,
    zIndex: 2
  },
  info: {
    paddingHorizontal: 40
  },
  avatarContainer: {
    position: "relative",
    marginTop: -80
  },
  avatar: {
    width: 124,
    height: 124,
    borderRadius: 62,
    borderWidth: 0
  },
  nameInfo: {
    marginTop: 35
  },
  divider: {
    width: "90%",
    borderWidth: 1,
    borderColor: "#E9ECEF"
  },
  thumb: {
    borderRadius: 4,
    marginVertical: 4,
    alignSelf: "center",
    width: thumbMeasure,
    height: thumbMeasure
  },
  text: { 
    fontFamily: 'C24',
    marginStart: 10,
    color: '#172B4D',
    // fontColor: '#172B4D',
  },
  text2: {
    fontFamily: 'NGB',
    marginTop: 3,
    marginEnd: 10,
  },
  textName: {
    fontFamily: 'SGB',
  },
  poptitle: {
    fontFamily: 'C24',
    marginBottom: 30,
    fontSize: 20,
  }
});

export default Profile;
