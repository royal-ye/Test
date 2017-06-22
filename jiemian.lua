----------------------------------------------------------------------------
-- Lua code generated with wxFormBuilder (version Jun 17 2015)
-- http://www.wxformbuilder.org/
----------------------------------------------------------------------------

-- Load the wxLua module, does nothing if running from wxLua, wxLuaFreeze, or wxLuaEdit
package.cpath = package.cpath..";./?.dll;./?.so;../lib/?.so;../lib/vc_dll/?.dll;../lib/bcc_dll/?.dll;../lib/mingw_dll/?.dll;"
require("wx")

UI = {}


-- create MyFrame
UI.MyFrame = wx.wxFrame (wx.NULL, wx.wxID_ANY, "", wx.wxDefaultPosition, wx.wxSize( 500,300 ), wx.wxDEFAULT_FRAME_STYLE+wx.wxTAB_TRAVERSAL )
	UI.MyFrame:SetSizeHints( wx.wxDefaultSize, wx.wxDefaultSize )
	
	UI.bSizer2 = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.m_button1 = wx.wxButton( UI.MyFrame, wx.wxID_ANY, "run", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.bSizer2:Add( UI.m_button1, 0, wx.wxALL, 5 )
	
	UI.m_staticText2 = wx.wxStaticText( UI.MyFrame, wx.wxID_ANY, "test_use", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText2:Wrap( -1 )
	UI.bSizer2:Add( UI.m_staticText2, 0, wx.wxALL, 5 )
	
	UI.text_main = wx.wxTextCtrl( UI.MyFrame, wx.wxID_ANY, "", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.bSizer2:Add( UI.text_main, 0, wx.wxALL, 5 )
	
	
	UI.MyFrame:SetSizer( UI.bSizer2 )
	UI.MyFrame:Layout()
	
	UI.MyFrame:Centre( wx.wxBOTH )
	
	-- Connect Events
	
	UI.m_button1:Connect( wx.wxEVT_COMMAND_BUTTON_CLICKED, function(event)
	--implements click_for_clear
	
	event:Skip()
	end )
	


--wx.wxGetApp():MainLoop()

